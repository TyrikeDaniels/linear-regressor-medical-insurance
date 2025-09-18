from pathlib import Path

import pandas as pd
from loguru import logger
import typer

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

from .config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

@app.command()
def process(
    input_path: Path = typer.Option(RAW_DATA_DIR, help="Raw data path."),
    output_path: Path =  typer.Option(PROCESSED_DATA_DIR, help="Download path."),
):
    """
    Pass raw data into pipeline.
    """

    raw_path = input_path / "insurance.csv"
    labels_path = output_path / "labels.csv"
    features_path = output_path / "features.csv"

    logger.info(f"Importing CSV from {raw_path} into DataFrame object.")
    df = pd.read_csv(raw_path)

    y = df.iloc[:, -1:]
    x = df.iloc[:, :-1]

    categorical_features = x.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = x.select_dtypes(include=['int64', 'float64']).columns.tolist()

    binary_features = [col for col in categorical_features if x[col].nunique() == 2]
    nominal = [col for col in categorical_features if x[col].nunique() > 2]

    binary_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()) # StandardScalar
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("binary", binary_transformer, binary_features),
        ("categorical", categorical_transformer, nominal),
        ("numeric", numeric_transformer, numerical_features)
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor)
    ])

    logger.info(f"Passing raw data through pipeline.")
    processed_x = pipeline.fit_transform(x)

    ohe_columns = pipeline.named_steps['preprocessor'] \
        .named_transformers_['categorical'] \
        .named_steps['onehot'].get_feature_names_out(nominal)
    all_columns = list(binary_features) + list(ohe_columns) + numerical_features

    temp_x = pd.DataFrame(processed_x, columns=all_columns)
    temp_y = y.reset_index(drop=True)

    logger.info(f"Processing labels and features into CSV file at {output_path}.")
    temp_x.to_csv(features_path, index=False)
    temp_y.to_csv(labels_path, index=False)
    logger.success(f"Processing complete.")

if __name__ == "__main__":
    app()
