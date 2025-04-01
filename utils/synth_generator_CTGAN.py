import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')


class OptimizedCTGAN:
    """
    Оптимизированный генератор синтетических данных на основе CTGAN.
    Использует актуальную структуру библиотеки SDV.
    """

    def __init__(self,
                 categorical_threshold=15,
                 epochs=100,
                 batch_size=100,
                 generator_dim=(128, 128),
                 discriminator_dim=(128, 128),
                 embedding_dim=64,
                 random_state=42,
                 cuda=False,
                 verbose=True):
        """
        Инициализация генератора

        Параметры:
        ----------
        categorical_threshold : int, default=15
            Макс. количество уникальных значений для определения категориальных признаков
        epochs : int, default=100
            Количество эпох обучения
        batch_size : int, default=100
            Размер батча для обучения
        generator_dim : tuple, default=(128, 128)
            Размеры скрытых слоев генератора
        discriminator_dim : tuple, default=(128, 128)
            Размеры скрытых слоев дискриминатора
        embedding_dim : int, default=64
            Размер эмбеддингов для категориальных признаков
        random_state : int, default=42
            Начальное значение для генератора случайных чисел
        cuda : bool, default=False
            Использовать ли CUDA GPU
        verbose : bool, default=True
            Выводить ли информацию о процессе обучения
        """
        self.categorical_threshold = categorical_threshold
        self.epochs = epochs
        self.batch_size = batch_size
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.embedding_dim = embedding_dim
        self.random_state = random_state
        self.cuda = cuda
        self.verbose = verbose

        self.categorical_columns = []
        self.continuous_columns = []
        self.encoders = {}
        self.metadata = None
        self.model = None

    def _identify_column_types(self, df):
        """Определение типов столбцов"""
        self.categorical_columns = []
        self.continuous_columns = []

        for col in df.columns:
            n_unique = df[col].nunique()
            if (df[col].dtype == 'object' or
                    pd.api.types.is_categorical_dtype(df[col]) or
                    (n_unique < self.categorical_threshold and n_unique > 1)):
                self.categorical_columns.append(col)
            else:
                self.continuous_columns.append(col)

        print(
            f"Определено {len(self.continuous_columns)} числовых и {len(self.categorical_columns)} категориальных признаков")

    def _prepare_data(self, df):
        """Подготовка данных для CTGAN"""
        df_processed = df.copy()

        # Обработка NaN в числовых столбцах
        for col in self.continuous_columns:
            if df_processed[col].isna().any():
                # Заполняем пропуски медианой
                median_value = df_processed[col].median()
                df_processed[col] = df_processed[col].fillna(median_value)

        # Кодируем категориальные признаки
        for col in self.categorical_columns:
            # Заполняем пропуски
            if df_processed[col].isna().any():
                df_processed[col] = df_processed[col].fillna('missing')

            # Преобразуем все значения в строки
            df_processed[col] = df_processed[col].astype(str)

        return df_processed

    def _create_metadata(self, df):
        """Создание метаданных для CTGAN"""
        print("Создание метаданных...")

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=df)

        # Принудительно указываем категориальные признаки для уверенности
        for col in self.categorical_columns:
            if col in metadata.columns:
                metadata.update_column(column_name=col, sdtype='categorical')

        return metadata

    def fit(self, df):
        """
        Обучение генератора на исходных данных

        Параметры:
        ----------
        df : pandas.DataFrame
            Исходный датафрейм для обучения
        """
        print(f"Начало обучения на данных размера {df.shape}")

        # Идентифицируем типы столбцов и подготавливаем данные
        self._identify_column_types(df)
        df_processed = self._prepare_data(df)

        # Создаем метаданные
        self.metadata = self._create_metadata(df_processed)

        # Создаем модель CTGAN с оптимизированными параметрами
        print(f"Инициализация CTGAN: epochs={self.epochs}, batch_size={self.batch_size}")
        self.model = CTGANSynthesizer(
            metadata=self.metadata,
            enforce_rounding=False,  # Отключаем принудительное округление
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            generator_dim=self.generator_dim,
            discriminator_dim=self.discriminator_dim,
            embedding_dim=self.embedding_dim,
            cuda=self.cuda
        )

        # Обучаем модель с обработкой ошибок и индикатором прогресса
        print("Начало обучения модели CTGAN...")
        try:
            self.model.fit(df_processed)
            print("Обучение CTGAN завершено успешно!")
            return self
        except Exception as e:
            print(f"Произошла ошибка при обучении CTGAN: {e}")
            # В случае ошибки - уменьшаем параметры и пробуем снова
            print("Пробуем с уменьшенными параметрами...")
            self.model = CTGANSynthesizer(
                metadata=self.metadata,
                enforce_rounding=False,
                epochs=self.epochs // 2,  # Уменьшаем эпохи
                batch_size=max(32, self.batch_size // 2),
                verbose=self.verbose,
                generator_dim=(64, 64),
                discriminator_dim=(64, 64),
                embedding_dim=32,
                cuda=self.cuda
            )
            try:
                self.model.fit(df_processed)
                print("Обучение CTGAN с уменьшенными параметрами завершено успешно!")
                return self
            except Exception as e2:
                print(f"Вторая попытка также завершилась ошибкой: {e2}")
                raise RuntimeError(
                    "Не удалось обучить CTGAN. Попробуйте уменьшить размер данных или использовать другой генератор.")

    def generate(self, count=None):
        """
        Генерация синтетических данных

        Параметры:
        ----------
        count : int, default=None
            Количество синтетических образцов

        Возвращает:
        ----------
        pandas.DataFrame
            Датафрейм с синтетическими данными
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод fit().")

        print(f"Генерация {count} синтетических образцов...")

        try:
            # Генерируем данные
            synthetic_data = self.model.sample(num_rows=count)

            # Постобработка данных: исправление типов для числовых столбцов
            for col in self.continuous_columns:
                if col in synthetic_data.columns:
                    if 'int' in str(synthetic_data[col].dtype) or any(
                            substr in col for substr in ['Year', 'SF', 'Cars', 'Rooms', 'Bath', 'Bed']):
                        synthetic_data[col] = synthetic_data[col].round().astype(int)

            print(f"Синтетические данные успешно сгенерированы: {synthetic_data.shape}")
            return synthetic_data

        except Exception as e:
            print(f"Ошибка при генерации данных: {e}")
            # Если генерация не удалась, можно попробовать генерировать меньшими батчами
            try:
                batch_size = count // 4 + 1
                print(f"Пробуем генерировать данные батчами по {batch_size}...")

                synthetic_batches = []
                for i in range(0, count, batch_size):
                    batch_count = min(batch_size, count - i)
                    batch = self.model.sample(num_rows=batch_count)
                    synthetic_batches.append(batch)

                synthetic_data = pd.concat(synthetic_batches, ignore_index=True)

                # Постобработка данных
                for col in self.continuous_columns:
                    if col in synthetic_data.columns:
                        if 'int' in str(synthetic_data[col].dtype) or any(
                                substr in col for substr in ['Year', 'SF', 'Cars', 'Rooms', 'Bath', 'Bed']):
                            synthetic_data[col] = synthetic_data[col].round().astype(int)

                print(f"Синтетические данные успешно сгенерированы батчами: {synthetic_data.shape}")
                return synthetic_data

            except Exception as e2:
                print(f"Генерация батчами также закончилась ошибкой: {e2}")
                raise RuntimeError("Не удалось сгенерировать синтетические данные с помощью CTGAN.")


# Пример использования:
"""
# Загрузка данных
df = pd.read_csv('your_data.csv')

# Убедитесь, что установлены необходимые пакеты:
# pip install sdv

# Инициализация и обучение CTGAN
ctgan_generator = OptimizedCTGAN(
    categorical_threshold=15,  # Порог для определения категориальных признаков
    epochs=100,                # Количество эпох
    batch_size=64,             # Размер батча
    generator_dim=(128, 128),  # Размеры генератора
    cuda=False,                # Установите True, если есть GPU
    verbose=True               # Показывать прогресс обучения
)

# Обучаем модель
ctgan_generator.fit(df)

# Генерируем синтетические данные
synthetic_data = ctgan_generator.generate(count=1500)

# Сохраняем результат
synthetic_data.to_csv('synthetic_data_ctgan.csv', index=False)

# Объединяем с исходными данными, если нужно
combined_data = pd.concat([df, synthetic_data], ignore_index=True)
print(f"Размер объединенных данных: {combined_data.shape}")
"""
