import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from catboost import CatBoostClassifier, CatBoostRegressor
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
