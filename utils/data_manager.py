import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class DataManager:
    """Класс-хелпер для работы с данными."""

    @staticmethod
    def get_all_nan_cols(df):
        """Возвращает список колонок, содержащих NaN"""
        return df.columns[df.isnull().all()].tolist()

    @staticmethod
    def split_data_set_to_x_y(data_set, y_column):
        """Разделяет датасет на признаки и целевую переменную"""
        ds_copy = data_set.copy()
        return ds_copy.drop(columns=[y_column]), ds_copy[y_column]

    @staticmethod
    def split_data_set_to_x_ids(data_set, id_column):
        """Разделяет датасет на признаки и id для submission"""
        ds_copy = data_set.copy()
        return ds_copy.drop(columns=[id_column]), ds_copy[id_column]

    @staticmethod
    def show_unique_values(df):
        """Выводит уникальные значения для всех колонок"""
        for col in df.columns:
            print(f"Колонка: {col}")
            print(f"Уникальные значения: {df[col].unique()}")
            print(f"Количество уникальных значений: {df[col].nunique()}")
            print("-" * 50)

    @staticmethod
    def analyze_missing_values(df):
        """Базовый анализ пропущенных значений"""
        # Общее количество пропущенных значений
        total_missing = df.isna().sum().sum()

        # Количество пропущенных значений по колонкам
        missing_by_column = df.isna().sum()

        # Процент пропущенных значений по колонкам
        missing_percentage = (df.isna().sum() / len(df) * 100).round(2)

        # Создание DataFrame с результатами
        missing_info = pd.DataFrame({
            'Количество пропусков': missing_by_column,
            'Процент пропусков (%)': missing_percentage
        })

        # Сортировка по количеству пропусков (по убыванию)
        missing_info = missing_info.sort_values('Количество пропусков', ascending=False)

        return missing_info

    @staticmethod
    def visualize_missing_values(df):
        """Визуализация пропущенных значений"""
        plt.figure(figsize=(12, 6))

        # Создание тепловой карты пропущенных значений
        sns.heatmap(df.isna(), cmap='viridis', cbar=False, yticklabels=False)
        plt.title('Тепловая карта пропущенных значений')
        plt.xlabel('Колонки')
        plt.ylabel('Строки')
        plt.tight_layout()
        plt.savefig('missing_values_heatmap.png')
        plt.show()

        # Создание гистограммы пропущенных значений по колонкам
        plt.figure(figsize=(12, 6))
        df.isna().sum().sort_values(ascending=False).plot(kind='bar')
        plt.title('Количество пропущенных значений по колонкам')
        plt.xlabel('Колонки')
        plt.ylabel('Количество пропущенных значений')
        plt.tight_layout()
        plt.savefig('missing_values_by_column.png')
        plt.show()

    @staticmethod
    def get_numeric_and_categorical_features(df):
        # Получение числовых колонок
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        # Получение нечисловых колонок (всех остальных)
        non_numeric_columns = df.select_dtypes(exclude=['float64', 'int64']).columns
        return numeric_columns, non_numeric_columns

    @staticmethod
    def make_mi_scores(X, y):
        """
        Calculate the mutual information scores between features and target variable.

        Существует специальная метрика “взаимная информация” (она же mutual information),
        которая обнаруживает любые типы связей между двумя величинами.
        Минимальный MI-score между двумя величинами — нуль. Если это так, то можно сказать, что две
        рассматриваемые переменные полностью независимы друг от друга. Верхней же границы у MI-score не
        существует, хотя, на практике, редко встречаются значения больше 2.

        MI-score показывает на каких фичах стоит сосредоточить внимание в процессе FE
        (самые топовые по скорингу, наиболее информативные), а какие фичи следует дропнуть
        (те, которые около нуля показываю информативность - если обучить модель с ними, есть риск получить
        оверфиттинг - ибо слишком уж специфичные фичи)
        """
        X = X.copy()
        for colname in X.select_dtypes(["object", "category"]):
            X[colname], _ = X[colname].factorize()
        # All discrete features should now have integer dtypes
        discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
        mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        return mi_scores

    @staticmethod
    def plot_mi_scores(scores):
        scores = scores.sort_values(ascending=True)
        width = np.arange(len(scores))
        ticks = list(scores.index)
        plt.barh(width, scores)
        plt.yticks(width, ticks)
        plt.title("Mutual Information Scores")

    @staticmethod
    def drop_uninformative(df, mi_scores):
        return df.loc[:, mi_scores > 0.0]

    @staticmethod
    def label_encode(df):
        """
        Label Encoding (преобразование категориальных признаков в числовой вид).

        Для всех столбцов датафрейма, у которых тип данных object или category.
        Функция упрощает процесс кодирования категориальных данных, а сохранённые энкодеры позволяют переиспользовать
        те же преобразования на новых данных.
        Словарь encoders нужен для:
            •	Чтобы сохранить соответствие между исходными значениями признаков и их числовым представлением.
            •	Чтобы применить те же преобразования к новым (тестовым или валидным) данным.

        Применение на новых данных:
        X_train_encoded, encoders = label_encode(X_train)
        # применение на тестовом наборе:
        X_test_encoded = X_test.copy()
        for colname, encoder in encoders.items():
            X_test_encoded[colname] = encoder.transform(X_test_encoded[colname])
        """
        X = df.copy()
        encoders = {}

        for colname in X.select_dtypes(["category", "object"]):
            encoders[colname] = LabelEncoder()
            X[colname] = encoders[colname].fit_transform(X[colname])

        return X, encoders  # возвращаем также словарь с энкодерами для использования на тестовых данных

    @staticmethod
    def ohe(df):
        X = df.copy()
        encoders = {}

        for colname in X.select_dtypes(["category", "object"]):
            encoders[colname] = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            X[colname] = encoders[colname].fit_transform(X[colname])

        return X, encoders  # возвращаем также словарь с энкодерами для использования на тестовых данных

    @staticmethod
    def corrplot(df, method="pearson", annot=True, **kwargs):
        """Correlation matrix for the dataset.
        Что показывает этот график?

        Это матрица корреляций, визуализированная с помощью иерархической кластеризации:
            •	Каждая клетка отражает корреляцию между двумя признаками в твоём наборе данных.
            •	Цвет ячейки показывает силу и направление корреляции:
            •	Синий цвет — отрицательная корреляция (одна величина увеличивается, другая уменьшается).
            •	Красный цвет — положительная корреляция (обе величины увеличиваются или уменьшаются одновременно).
            •	Белый цвет — корреляция близка к 0 (слабая связь между признаками).
            •	Чем насыщеннее цвет (красный или синий), тем выше сила связи между признаками.
            •	Главная диагональ всегда имеет идеальную корреляцию (1.0), поскольку каждый признак полностью коррелирует сам с собой.

        По осям графика (сверху и слева) есть древовидные структуры — это дендрограммы. Они отражают иерархическую кластеризацию признаков:
            •	Похожие признаки группируются рядом.
            •	Чем ближе признаки на дендрограмме, тем сильнее они связаны друг с другом и имеют похожий паттерн корреляций с другими признаками.
            •	Эта группировка помогает понять структуру и взаимосвязи внутри данных.

        Как извлечь смысл из этого графика?

        Для анализа датасета по прогнозу цен на дома (например, Kaggle House Prices):
            1.	Определение групп признаков
            •	Ищи группы признаков, объединённые рядом на дендрограмме.
            •	Группировка подскажет тебе, какие признаки стоит рассматривать вместе, а какие дублируют друг друга и могут быть избыточными.
            2.	Поиск сильных корреляций с целевым признаком
            •	Например, если твоя целевая переменная – цена дома (SalePrice), найди её в списке признаков и посмотри, какие признаки имеют с ней ярко выраженную корреляцию (красные или синие ячейки, удалённые от белого цвета).
            •	Высококоррелирующие признаки с ценой дома могут быть важными для предсказательной модели.
            3.	Выявление избыточных признаков (мультиколлинеарность)
            •	Если признаки очень сильно коррелируют между собой (например, признаки площади гаража и количество машин в гараже), есть смысл использовать один из признаков, чтобы не было мультиколлинеарности в моделях (особенно важно для линейных моделей).
            4.	Удаление слабых признаков
            •	Если признаки почти не имеют корреляции ни с ценой, ни с другими признаками (белые клетки), они могут быть кандидатами на удаление.

        Как работать дальше:
            •	После анализа графика:
            1.	Отметь признаки с сильной корреляцией с целевой переменной.
            2.	Исключи признаки, которые слишком сильно коррелируют друг с другом (оставь один представитель каждой группы).
            3.	Проверь влияние этих решений на качество модели.
            4. Группы сильно коррелированных признаков часто дают интересные нагрузки для PCA (Principal Component Analysis) и LDA (Linear Discriminant Analysis).
        """
        sns.clustermap(
            df.corr(method, numeric_only=True),
            vmin=-1.0,
            vmax=1.0,
            cmap="icefire",
            method="complete",
            annot=annot,
            **kwargs,
        )


class CrossFoldEncoder:
    """Cross-validation encoder for categorical features.

    Позволяет применять любой тип энкодера (например, TargetEncoder) с кросс-валидацией. CV тут нужно для того,
    чтобы не терять данные. Т.е. если мы делаем TargetEncoder, то в тестовой выборке не должно быть целевой переменной.
    Поэтому мы делаем кросс-валидацию, чтобы в каждом из фолдов была своя целевая переменная.
    В этом случае мы можем использовать TargetEncoder, но при этом не теряем данные.

    Use it like:
    encoder = CrossFoldEncoder(MEstimateEncoder, m=1)
    X_encoded = encoder.fit_transform(X, y, cols=["MSSubClass"]))
    You can turn any of the encoders from the category_encoders library into a cross-fold encoder.
    The CatBoostEncoder would be worth trying. It's similar to MEstimateEncoder but uses some tricks to better
    prevent overfitting. Its smoothing parameter is called a instead of m.
    """
    def __init__(self, encoder, **kwargs):
        self.encoder_ = encoder
        self.kwargs_ = kwargs  # keyword arguments for the encoder
        self.cv_ = KFold(n_splits=5)

    # Fit an encoder on one split and transform the feature on the
    # other. Iterating over the splits in all folds gives a complete
    # transformation. We also now have one trained encoder on each
    # fold.
    def fit_transform(self, X, y, cols):
        self.fitted_encoders_ = []
        self.cols_ = cols
        X_encoded = []
        for idx_encode, idx_train in self.cv_.split(X):
            fitted_encoder = self.encoder_(cols=cols, **self.kwargs_)
            fitted_encoder.fit(
                X.iloc[idx_encode, :], y.iloc[idx_encode],
            )
            X_encoded.append(fitted_encoder.transform(X.iloc[idx_train, :])[cols])
            self.fitted_encoders_.append(fitted_encoder)
        X_encoded = pd.concat(X_encoded)
        X_encoded.columns = [name + "_encoded" for name in X_encoded.columns]
        return X_encoded

    # To transform the test data, average the encodings learned from
    # each fold.
    def transform(self, X):
        from functools import reduce

        X_encoded_list = []
        for fitted_encoder in self.fitted_encoders_:
            X_encoded = fitted_encoder.transform(X)
            X_encoded_list.append(X_encoded[self.cols_])
        X_encoded = reduce(
            lambda x, y: x.add(y, fill_value=0), X_encoded_list
        ) / len(X_encoded_list)
        X_encoded.columns = [name + "_encoded" for name in X_encoded.columns]
        return X_encoded
