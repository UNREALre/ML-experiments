import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class ModelManager:
    """Класс-хелпер для работы с моделями."""

    @staticmethod
    def save_model(model, model_path):
        """Сохраняет модель в файл."""
        joblib.dump(model, model_path)

    @staticmethod
    def load_model(model_path):
        """Загружает модель из файла."""
        return joblib.load(model_path)

    @staticmethod
    def get_top_features(model_pipeline, n=10, importance_type='default', return_names_only=True):
        """
        Возвращает топ-N наиболее важных фич для различных типов моделей.

        Параметры:
        ----------
        model_pipeline : Pipeline
            Sklearn pipeline с препроцессором и моделью
        n : int, optional (default=10)
            Количество возвращаемых важных фич
        importance_type : str, optional (default='default')
            Тип важности для CatBoost и XGBoost моделей
        return_names_only : bool, optional (default=True)
            Если True, возвращает только список имен фич
            Если False, возвращает DataFrame с именами и важностями

        Возвращает:
        ----------
        list или pandas.DataFrame : список имен фич или таблица с важностями фич
        """
        try:
            # Получаем имена фич после препроцессинга
            feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
        except:
            # Если нет препроцессора или другая структура pipeline
            try:
                feature_names = model_pipeline.feature_names_in_
            except:
                raise ValueError("Не удалось получить имена фич. Проверьте структуру pipeline.")

        # Получаем модель
        model = model_pipeline.named_steps['model']

        # Определяем тип модели и получаем важности признаков
        if hasattr(model, 'feature_importances_'):
            # RandomForest, GradientBoosting, LightGBM, XGBoost (базовый случай)
            importances = model.feature_importances_
            importance_name = 'Feature Importance'

        elif hasattr(model, 'coef_'):
            # Линейные модели
            if len(model.coef_.shape) > 1:
                # Для многоклассовой классификации берем среднее по классам
                importances = np.abs(model.coef_).mean(axis=0)
            else:
                importances = np.abs(model.coef_)
            importance_name = 'Coefficient Magnitude'

        elif isinstance(model, (CatBoostClassifier, CatBoostRegressor)):
            # CatBoost модели
            if importance_type == 'default':
                # По умолчанию используем PredictionValuesChange
                importances = model.get_feature_importance()
            else:
                # Используем указанный тип важности
                importances = model.get_feature_importance(type=importance_type)
            importance_name = f'CatBoost Importance ({importance_type})'

        elif isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor)):
            # XGBoost модели с дополнительными типами важности
            if importance_type != 'default':
                importances = model.get_booster().get_score(importance_type=importance_type)
                # Преобразуем словарь в массив, сохраняя порядок фич
                importances_array = np.zeros(len(feature_names))
                for key, value in importances.items():
                    try:
                        # XGBoost может использовать f0, f1, ... вместо имен
                        if key.startswith('f'):
                            idx = int(key[1:])
                            importances_array[idx] = value
                    except:
                        pass
                importances = importances_array
            else:
                importances = model.feature_importances_
            importance_name = f'XGBoost Importance ({importance_type})'

        else:
            raise ValueError(f"Модель типа {type(model)} не поддерживается.")

        # Проверка на соответствие размерности
        if len(importances) != len(feature_names):
            raise ValueError(
                f"Количество фич ({len(feature_names)}) не соответствует количеству важностей ({len(importances)})")

        # Создаем DataFrame с результатами
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })

        # Сортируем по убыванию важности
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

        # Возвращаем топ-N фич
        if return_names_only:
            # Возвращаем только имена фич в виде списка
            return feature_importance_df.head(n)['Feature'].tolist()
        else:
            # Возвращаем DataFrame с именами и важностями
            return feature_importance_df.head(n).reset_index(drop=True)

    @staticmethod
    def train_evaluate_catboost_model(X_train,
                                      X_test,
                                      y_train,
                                      y_test,
                                      features,
                                      model_params=None,
                                      random_state=42
                                      ):
        # Выбираем только заданные признаки
        X_train_sel = X_train[features].copy()
        X_test_sel = X_test[features].copy()

        # Определяем типы признаков динамически
        numeric_cols = X_train_sel.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_cols = X_train_sel.select_dtypes(exclude=['float64', 'int64']).columns.tolist()

        # Создаём новый препроцессор, соответствующий текущим признакам
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numeric_cols),

                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_cols)
            ],
            sparse_threshold=0,
            remainder='drop'
        )

        # Если гиперпараметры не заданы, используем значения по умолчанию
        if model_params is None:
            model_params = {
                'iterations': 1000,
                'learning_rate': 0.1,
                'depth': 6,
                'loss_function': 'RMSE',
                'verbose': 100,
                'random_seed': random_state,
            }

        # Пайплайн
        pipeline = Pipeline([
            ('preprocessor', preprocessor),  # предполагается, что preprocessor уже настроен
            ('model', CatBoostRegressor(**model_params))
        ])

        # Тренировка
        pipeline.fit(X_train_sel, y_train)

        # Предсказания
        y_pred = pipeline.predict(X_test_sel)

        # Метрики
        mse = mean_squared_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mean_target = np.mean(y_test)

        return {'mse': mse, 'rmse': rmse, 'rmse %': rmse/mean_target*100, 'r2': r2, 'model': pipeline}

    @staticmethod
    def evaluate_regression(model, x_val, y_val):
        """
        Вычисляет, выводит и возвращает набор стандартных метрик регрессии.

        Args:
            model: Обученная модель с методом .predict().
            x_val (pd.DataFrame или np.ndarray): Валидационные признаки.
            y_val (pd.Series или np.ndarray): Истинные значения для валидации.

        Returns:
            dict: Словарь с названиями метрик и их значениями.
        """
        # 1. Получаем предсказания
        y_pred = model.predict(x_val)

        # 2. Рассчитываем метрики из sklearn.metrics
        r2 = metrics.r2_score(y_val, y_pred)
        mse = metrics.mean_squared_error(y_val, y_pred)
        mae = metrics.mean_absolute_error(y_val, y_pred)

        # RMSE - это корень из MSE
        rmse = np.sqrt(mse)
        # Альтернативно: rmse = metrics.mean_squared_error(y_val, y_pred, squared=False)

        # --- Метрики, чувствительные к значениям ---
        # MSLE и RMSLE требуют, чтобы все значения y_val и y_pred были неотрицательными.
        # MAPE требует, чтобы y_val не содержал нулей.

        msle = np.nan  # Инициализируем как NaN на случай ошибки
        rmsle_score = np.nan
        mape = np.nan

        # Проверка на отрицательные значения в y_val (для MSLE/RMSLE)
        if np.any(y_val < 0):
            print("Предупреждение: Обнаружены отрицательные значения в y_val. MSLE и RMSLE не могут быть вычислены.")
        else:
            # Некоторые модели могут предсказывать отрицательные значения, даже если y_val положительный.
            # Для MSLE/RMSLE заменим отрицательные предсказания на 0.
            y_pred_non_negative = np.maximum(y_pred, 0)
            try:
                msle = metrics.mean_squared_log_error(y_val, y_pred_non_negative)
                rmsle_score = np.sqrt(msle)
            except ValueError as e:
                print(f"Ошибка при расчете MSLE/RMSLE: {e}")

        # Проверка на нули в y_val (для MAPE)
        if np.any(y_val == 0):
            print(
                "Предупреждение: Обнаружены нули в y_val. MAPE может быть неточным или вызвать ошибку. Расчет для ненулевых значений...")
            # Рассчитываем MAPE только для ненулевых значений y_val
            mask = y_val != 0
            if np.any(mask):  # Если есть хоть какие-то ненулевые значения
                mape = metrics.mean_absolute_percentage_error(y_val[mask], y_pred[mask])
            # else: mape остается np.nan
        else:
            # Если нулей нет, считаем как обычно
            mape = metrics.mean_absolute_percentage_error(y_val, y_pred)

        # 3. Выводим результаты
        print(f"R2 Score: {r2:.4f}")  # Форматируем для читаемости
        print(f"MSE:      {mse:.4f}")
        print(f"RMSE:     {rmse:.4f}")
        print(f"MAE:      {mae:.4f}")
        if not np.isnan(msle):
            print(f"MSLE:     {msle:.4f}")
        if not np.isnan(rmsle_score):
            print(f"RMSLE:    {rmsle_score:.4f}")  # Корень из MSLE
        if not np.isnan(mape):
            print(f"MAPE:     {mape:.4f}")  # MAPE возвращается как доля, можно умножить на 100 для процентов

        # 4. Возвращаем словарь
        results = {
            "R2": r2,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "MSLE": msle,
            "RMSLE": rmsle_score,
            "MAPE": mape
        }
        return results
