import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

try:
    from sdv.single_table import GaussianCopulaSynthesizer

    GAUSSIAN_COPULA_AVAILABLE = True
except ImportError:
    GAUSSIAN_COPULA_AVAILABLE = False
    print("Библиотека SDV не установлена. Для использования GaussianCopula установите: pip install sdv")


class CombinedSyntheticGenerator:
    """
    Комбинированный генератор синтетических данных, использующий:
    1. SMOTE для генерации числовых данных
    2. GaussianCopula (если доступно) для сохранения распределения и корреляций
    """

    def __init__(self, categorical_threshold=20, random_state=42,
                 use_gaussian_copula=True, max_clusters=10):
        """
        Инициализация генератора

        Параметры:
        ----------
        categorical_threshold : int, default=20
            Максимальное количество уникальных значений для определения категориальных признаков
        random_state : int, default=42
            Начальное значение для генератора случайных чисел
        use_gaussian_copula : bool, default=True
            Использовать ли GaussianCopula (если доступно)
        max_clusters : int, default=10
            Максимальное количество кластеров для стратегии генерации
        """
        self.random_state = random_state
        self.categorical_threshold = categorical_threshold
        self.cat_cols = []
        self.num_cols = []
        self.encoders = {}
        self.imputers = {}
        self.scaler = None
        self.gaussian_copula = None
        self.use_gaussian_copula = use_gaussian_copula and GAUSSIAN_COPULA_AVAILABLE
        self.kmeans = None
        self.max_clusters = max_clusters
        self.df_preprocessed = None
        np.random.seed(random_state)

    def _identify_column_types(self, df):
        """Определение типов столбцов"""
        self.cat_cols = []
        self.num_cols = []

        for col in df.columns:
            # Определяем категориальные столбцы
            if (df[col].dtype == 'object' or
                    pd.api.types.is_categorical_dtype(df[col]) or
                    df[col].nunique() < self.categorical_threshold):
                self.cat_cols.append(col)
            else:
                self.num_cols.append(col)

        print(f"Определено {len(self.num_cols)} числовых и {len(self.cat_cols)} категориальных признаков")

    def _preprocess_data(self, df):
        """Предобработка данных для обучения"""
        df_processed = df.copy()

        # Обработка категориальных признаков
        for col in self.cat_cols:
            # Создаем импутер для заполнения пропусков
            imputer = SimpleImputer(strategy='most_frequent')
            # Получаем значения после импутирования
            values = imputer.fit_transform(df_processed[[col]])

            # Безопасно обрабатываем размерность
            if isinstance(values, np.ndarray) and values.ndim > 1 and values.shape[1] == 1:
                df_processed[col] = values.flatten()
            else:
                df_processed[col] = values

            self.imputers[col] = imputer

            # Преобразуем категориальные признаки в строки
            df_processed[col] = df_processed[col].astype(str)

        # Обработка числовых признаков
        if self.num_cols:
            # Импутер для числовых признаков
            num_imputer = SimpleImputer(strategy='median')
            df_processed[self.num_cols] = num_imputer.fit_transform(df_processed[self.num_cols])
            self.imputers['numerical'] = num_imputer

            # Масштабирование числовых признаков
            self.scaler = StandardScaler()
            df_processed[self.num_cols] = self.scaler.fit_transform(df_processed[self.num_cols])

        return df_processed

    def fit(self, df):
        """
        Обучение генератора на исходных данных

        Параметры:
        ----------
        df : pandas.DataFrame
            Исходный датафрейм для обучения
        """
        print(f"Начало обучения на данных размера {df.shape}")

        # Идентифицируем типы столбцов
        self._identify_column_types(df)

        # Предобработка данных
        self.df_preprocessed = self._preprocess_data(df)

        # Определяем оптимальное количество кластеров с помощью метода локтя
        if len(self.num_cols) > 0:
            inertia = []
            X_num = self.df_preprocessed[self.num_cols].values
            k_range = range(1, min(self.max_clusters + 1, len(df) // 10 + 1))

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state)
                kmeans.fit(X_num)
                inertia.append(kmeans.inertia_)

            # Находим точку локтя (можно улучшить этот алгоритм)
            deltas = np.diff(inertia)
            deltas_of_deltas = np.diff(deltas)
            elbow_idx = np.argmax(deltas_of_deltas) + 1 if len(deltas_of_deltas) > 0 else 2
            n_clusters = min(k_range[elbow_idx], self.max_clusters)

            # Обучаем итоговую модель кластеризации
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
            self.kmeans.fit(X_num)

            print(f"Определено оптимальное количество кластеров: {n_clusters}")

        # Обучаем GaussianCopula, если доступно
        if self.use_gaussian_copula:
            print("Обучение GaussianCopula...")
            try:
                # Возвращаем данные к исходным масштабам для GaussianCopula
                df_for_copula = df.copy()
                self.gaussian_copula = GaussianCopulaSynthesizer(
                    primary_key=None,
                    enforce_min_max_values=True,
                    default_distribution='gaussian'
                )
                self.gaussian_copula.fit(df_for_copula)
                print("GaussianCopula успешно обучена")
            except Exception as e:
                print(f"Ошибка при обучении GaussianCopula: {e}")
                self.use_gaussian_copula = False

        print("Обучение генератора завершено успешно")
        return self

    def generate(self, count=None):
        """
        Генерация синтетических данных

        Параметры:
        ----------
        count : int, default=None
            Количество синтетических образцов. Если None, используется размер
            исходного датасета.

        Возвращает:
        ----------
        pandas.DataFrame
            Датафрейм с синтетическими данными
        """
        if count is None:
            count = len(self.df_preprocessed)

        print(f"Генерация {count} синтетических образцов...")

        # Пробуем сначала с GaussianCopula, если доступно
        if self.use_gaussian_copula and self.gaussian_copula:
            try:
                print("Используем GaussianCopula для генерации...")
                synthetic_df = self.gaussian_copula.sample(count)
                print("Данные успешно сгенерированы с помощью GaussianCopula")
                return synthetic_df
            except Exception as e:
                print(f"Ошибка при генерации с помощью GaussianCopula: {e}")
                print("Переключаемся на генерацию с помощью SMOTE...")

        # Если GaussianCopula не сработала или не используется, применяем SMOTE
        if not self.num_cols:
            raise ValueError("Для использования SMOTE необходимо иметь числовые признаки")

        # Подготовка данных для SMOTE
        X = self.df_preprocessed[self.num_cols].values

        # Генерируем синтетический датасет на основе кластеризации и SMOTE
        if self.kmeans:
            # Получаем метки кластеров
            clusters = self.kmeans.predict(X)
            unique_clusters = np.unique(clusters)

            # Создаем искусственную целевую переменную на основе кластеров
            y = clusters

            # Инициализируем SMOTE
            smote = SMOTE(
                sampling_strategy='auto',
                random_state=self.random_state,
                k_neighbors=min(5, min(np.bincount(y)) - 1)  # Адаптивный k для малых кластеров
            )

            # Генерируем синтетические числовые данные
            X_synthetic, y_synthetic = smote.fit_resample(X, y)

            # Выбираем нужное количество образцов из сгенерированных
            indices = np.random.choice(len(X_synthetic), size=count, replace=len(X_synthetic) < count)
            X_final = X_synthetic[indices]
            cluster_labels = y_synthetic[indices]
        else:
            # Если кластеризация не использовалась, создаем заглушку для целевой переменной
            y = np.zeros(len(X))

            # Инициализируем SMOTE
            smote = SMOTE(random_state=self.random_state)

            # Генерируем синтетические числовые данные
            X_final, _ = smote.fit_resample(X, y)

            # Выбираем нужное количество образцов
            indices = np.random.choice(len(X_final), size=count, replace=len(X_final) < count)
            X_final = X_final[indices]
            cluster_labels = np.zeros(count)

        # Создаем датафрейм из числовых данных
        synthetic_df = pd.DataFrame(X_final, columns=self.num_cols)

        # Обратное масштабирование числовых признаков
        if self.scaler:
            synthetic_df[self.num_cols] = self.scaler.inverse_transform(synthetic_df[self.num_cols])

        # Генерируем категориальные признаки
        if self.cat_cols:
            # Группируем оригинальные данные по кластерам
            df_with_clusters = self.df_preprocessed.copy()
            if self.kmeans:
                df_with_clusters['cluster'] = clusters

            # Для каждого кластера сохраняем распределение категориальных признаков
            for col in self.cat_cols:
                if self.kmeans:
                    # Отбираем категории с учетом принадлежности к кластеру
                    synthetic_categories = []
                    for cluster in cluster_labels:
                        # Выбираем случайную категорию из соответствующего кластера
                        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster][col]
                        if len(cluster_data) > 0:
                            # Вероятностный выбор категории
                            category_probs = cluster_data.value_counts(normalize=True)
                            synthetic_categories.append(np.random.choice(
                                category_probs.index, p=category_probs.values
                            ))
                        else:
                            # Если кластер пустой, выбираем из всех данных
                            category_probs = self.df_preprocessed[col].value_counts(normalize=True)
                            synthetic_categories.append(np.random.choice(
                                category_probs.index, p=category_probs.values
                            ))
                else:
                    # Если кластеризация не использовалась, выбираем из всего распределения
                    category_probs = self.df_preprocessed[col].value_counts(normalize=True)
                    synthetic_categories = np.random.choice(
                        category_probs.index, size=count, p=category_probs.values
                    )

                synthetic_df[col] = synthetic_categories

        # Финальная обработка типов данных
        for col in self.num_cols:
            # Если в имени столбца есть подсказки о типе данных
            if ('Year' in col or 'SF' in col or 'Cars' in col or
                    col.endswith('Bldg') or 'Count' in col or col.endswith('Bath') or
                    col.endswith('AbvGr')):
                synthetic_df[col] = synthetic_df[col].round().astype(int)
            elif 'int' in str(synthetic_df[col].dtype):
                synthetic_df[col] = synthetic_df[col].round().astype(int)

        print(f"Синтетические данные успешно сгенерированы: {synthetic_df.shape}")
        return synthetic_df


# Пример использования:
'''
# Загрузка данных
df = pd.read_csv('your_data.csv')

# Инициализация и обучение генератора
generator = CombinedSyntheticGenerator(use_gaussian_copula=True)
generator.fit(df)

# Генерация синтетических данных
synthetic_df = generator.generate(count=1500)

# Сохранение результатов
synthetic_df.to_csv('synthetic_data.csv', index=False)
'''
