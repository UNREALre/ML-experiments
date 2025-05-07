import json


def save_features(df_or_array):
    """Сохраняет фичи ДФ в файл. Можно использовать потом как артефакт в MLflow."""
    path_to_features = "features.json"

    # Проверяем тип входных данных
    if hasattr(df_or_array, 'columns'):
        # Это pandas DataFrame
        features = df_or_array.columns.tolist()
    else:
        # Это numpy array, просто используем индексы как имена колонок
        features = [f"feature_{i}" for i in range(df_or_array.shape[1])]

    with open(path_to_features, "w") as f:
        json.dump(features, f)

    return path_to_features


def save_data(X, y, random_state=42):
    """Сохраняет данные в CSV файл. Можно использовать потом как артефакт в MLflow."""
    sample_df = (
        X
        .assign(target=y)
        .sample(frac=0.01, random_state=random_state)  # 1% данных
    )
    sample_file_path = "train_sample.csv"
    sample_df.to_csv(sample_file_path, index=False)

    return sample_file_path
