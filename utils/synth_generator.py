"""Модуль для генерации синтетических данных."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KernelDensity
import warnings

warnings.filterwarnings('ignore')


class SimpleSyntheticGenerator:
    """
    Простой и эффективный генератор синтетических данных,
    использующий комбинацию PCA и KDE (Kernel Density Estimation)
    """

    def __init__(self, categorical_max_unique=20, random_state=42):
        """
        Инициализация генератора

        Параметры:
        ----------
        categorical_max_unique : int, default=20
            Максимальное количество уникальных значений для автоматического определения
            категориальных признаков
        random_state : int, default=42
            Начальное значение для генератора случайных чисел
        """
        self.random_state = random_state
        self.categorical_max_unique = categorical_max_unique
        self.cat_cols = []
        self.num_cols = []
        self.cat_encoders = {}
        self.num_scaler = None
        self.num_imputer = None
        self.cat_imputer = None
        self.kde = None
        self.pca = None
        self.pca_components = None
        self.original_shape = None
        np.random.seed(random_state)

    def _identify_column_types(self, df):
        """Определение типов столбцов"""
        self.cat_cols = []
        self.num_cols = []

        for col in df.columns:
            # Определяем категориальные столбцы
            if df[col].dtype == 'object' or df[col].nunique() < self.categorical_max_unique:
                self.cat_cols.append(col)
            else:
                self.num_cols.append(col)

        print(f"Определено {len(self.num_cols)} числовых и {len(self.cat_cols)} категориальных признаков")

    def fit(self, df):
        """
        Обучение генератора на исходных данных

        Параметры:
        ----------
        df : pandas.DataFrame
            Исходный датафрейм для обучения
        """
        self.original_shape = df.shape
        print(f"Начало обучения на данных размера {self.original_shape}")

        # Идентифицируем типы столбцов
        self._identify_column_types(df)

        # Обработка числовых признаков
        if self.num_cols:
            # Создаем и обучаем импутер для числовых данных
            self.num_imputer = SimpleImputer(strategy='median')
            num_data_imputed = self.num_imputer.fit_transform(df[self.num_cols])

            # Нормализуем числовые данные
            self.num_scaler = StandardScaler()
            num_data_scaled = self.num_scaler.fit_transform(num_data_imputed)

            # Сохраняем обработанные числовые данные
            num_df = pd.DataFrame(num_data_scaled, columns=self.num_cols)
        else:
            num_df = pd.DataFrame()

        # Обработка категориальных признаков
        cat_df = pd.DataFrame()
        if self.cat_cols:
            # Создаем и обучаем импутер для категориальных данных
            self.cat_imputer = SimpleImputer(strategy='most_frequent')
            cat_data_imputed = self.cat_imputer.fit_transform(df[self.cat_cols])
            cat_df_imputed = pd.DataFrame(cat_data_imputed, columns=self.cat_cols)

            # Кодируем каждый категориальный столбец отдельно
            for col in self.cat_cols:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(cat_df_imputed[[col]])
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=[f"{col}_{val}" for val in encoder.categories_[0]]
                )
                cat_df = pd.concat([cat_df, encoded_df], axis=1)
                self.cat_encoders[col] = encoder

        # Объединяем числовые и закодированные категориальные данные
        transformed_df = pd.concat([num_df, cat_df], axis=1)
        print(f"Размер преобразованных данных: {transformed_df.shape}")

        # Применяем PCA для уменьшения размерности данных
        n_components = min(transformed_df.shape[1], 50)  # Максимум 50 компонент
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        self.pca_components = self.pca.fit_transform(transformed_df)

        explained_variance = sum(self.pca.explained_variance_ratio_) * 100
        print(f"PCA сохраняет {explained_variance:.2f}% информации с использованием {n_components} компонент")

        # Оцениваем плотность вероятности с помощью KDE
        self.kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
        self.kde.fit(self.pca_components)

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
            count = self.original_shape[0]

        print(f"Генерация {count} синтетических образцов...")

        # Генерируем выборки из распределения в пространстве PCA
        synthetic_pca = self.kde.sample(count, random_state=self.random_state)

        # Восстанавливаем оригинальное пространство
        synthetic_features = self.pca.inverse_transform(synthetic_pca)

        # Разделяем данные обратно на числовые и категориальные
        result_df = pd.DataFrame()

        # Обработка числовых признаков
        if self.num_cols:
            num_features_count = len(self.num_cols)
            num_synthetic = synthetic_features[:, :num_features_count]

            # Восстанавливаем масштаб числовых признаков
            num_synthetic_original_scale = self.num_scaler.inverse_transform(num_synthetic)

            # Добавляем числовые признаки в результат
            num_df = pd.DataFrame(num_synthetic_original_scale, columns=self.num_cols)
            result_df = pd.concat([result_df, num_df], axis=1)

        # Обработка категориальных признаков
        if self.cat_cols:
            start_idx = len(self.num_cols) if self.num_cols else 0

            for col in self.cat_cols:
                encoder = self.cat_encoders[col]
                n_categories = len(encoder.categories_[0])

                # Извлекаем One-Hot кодированную часть для текущего признака
                cat_synthetic = synthetic_features[:, start_idx:start_idx + n_categories]

                # Преобразуем вероятности в четкие категории
                cat_synthetic_prob = np.exp(cat_synthetic) / np.sum(np.exp(cat_synthetic), axis=1, keepdims=True)
                cat_indices = np.array([np.random.choice(n_categories, p=probs) for probs in cat_synthetic_prob])

                # Получаем оригинальные категории
                cat_values = encoder.categories_[0][cat_indices]

                # Добавляем категориальный признак в результат
                result_df[col] = cat_values

                start_idx += n_categories

        # Финальная обработка типов данных
        for col in self.num_cols:
            if 'int' in str(result_df[col].dtype) or col.endswith('SF') or col.endswith('Cars') or 'Year' in col:
                result_df[col] = result_df[col].round().astype(int)

        print(f"Синтетические данные успешно сгенерированы: {result_df.shape}")
        return result_df


# Пример использования:
'''
# Загрузка данных
df = pd.read_csv('your_data.csv')

# Инициализация и обучение генератора
generator = SimpleSyntheticGenerator()
generator.fit(df)

# Генерация синтетических данных
synthetic_df = generator.generate(count=1500)

# Сохранение результатов
synthetic_df.to_csv('synthetic_data.csv', index=False)
'''
