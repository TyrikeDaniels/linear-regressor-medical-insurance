from pathlib import Path

import pandas as pd
from loguru import logger
import typer

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

from lin_mod.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

@app.command()
def process(
        input_path: Path = typer.Option(RAW_DATA_DIR, help="Raw data path"),
        output_path: Path =  typer.Option(PROCESSED_DATA_DIR, help="Download path"),
):
    """Pass cleaned data into pipeline."""

    csv_file_input = input_path / "insurance.csv"
    csv_file_output = output_path / "insurance.csv"

    logger.info(f"Importing CSV from {csv_file_input} into DataFrame object...")
    df = pd.read_csv(csv_file_input)

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

    logger.info(f"Passing raw data through pipeline...")
    processed_x = pipeline.fit_transform(x)

    ohe_columns = pipeline.named_steps['preprocessor'] \
        .named_transformers_['categorical'] \
        .named_steps['onehot'].get_feature_names_out(nominal)
    all_columns = list(binary_features) + list(ohe_columns) + numerical_features

    temp_x = pd.DataFrame(processed_x, columns=all_columns)
    temp_y = y.reset_index(drop=True)

    processed_df = pd.concat([temp_x, temp_y], axis=1)

    processed_df.to_csv(csv_file_output)
    logger.success(f"Processing complete. Processed data has been sent to {csv_file_output}.")

if __name__ == "__main__":
    app()
