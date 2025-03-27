import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class DataManager:
    """Класс-хелпер для работы с данными."""

    @staticmethod
    def get_all_nan_cols(df):
        """Возвращает список колонок, содержащих NaN"""
        return df.columns[df.isnull().all()].tolist()

    @staticmethod
    def split_data_set_to_x_y(data_set, y_column):
        """Разделяет датасет на признаки и целевую переменную"""
        return data_set.drop(columns=[y_column]), data_set[y_column]

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