"""Train a used-car selling-price model using the real CarDekho CSV."""

from pathlib import Path
import sys

try:
    import joblib
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
except ImportError as exc:
    print(
        f"Missing dependency: {exc.name}. Activate the project environment and "
        "run `python -m pip install -r requirements.txt`.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "submission"
TARGET = "Selling_Price"


def load_dataset(data_dir: Path | None = None) -> pd.DataFrame:
    """Load a CSV from data/ only if it contains the expected target column."""
    dataset_dir = DATA_DIR if data_dir is None else Path(data_dir)
    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV dataset found in {dataset_dir}. Follow data/README.md to obtain "
            "the real CarDekho data and place it there."
        )

    for csv_path in csv_files:
        try:
            frame = pd.read_csv(csv_path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            print(f"Could not read {csv_path.name}: {exc}", file=sys.stderr)
            continue
        if TARGET in frame.columns:
            print(f"Using dataset: {csv_path}")
            return frame

    raise ValueError(
        f"None of the CSV files in {dataset_dir} contains the required "
        f"`{TARGET}` column. See data/README.md for the expected CarDekho file."
    )


def prepare_dataset(cars: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and rows without a numeric target."""
    if TARGET not in cars.columns:
        raise ValueError(f"Dataset must contain the target column `{TARGET}`.")

    prepared = cars.drop_duplicates().copy()
    prepared[TARGET] = pd.to_numeric(prepared[TARGET], errors="coerce")
    prepared = prepared.dropna(subset=[TARGET]).copy()
    if len(prepared) < 2:
        raise ValueError("At least two rows with a numeric Selling_Price are required.")
    return prepared


def split_dataset(
    cars: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split prepared data into a reproducible 80% train and 20% test set."""
    features = cars.drop(columns=[TARGET])
    target = cars[TARGET]
    return train_test_split(features, target, test_size=0.2, random_state=42)


def build_model(features: pd.DataFrame) -> Pipeline:
    """Build a preprocessing and regression pipeline fitted only on training data."""
    numeric_features = features.select_dtypes(include="number").columns.tolist()
    categorical_features = features.select_dtypes(exclude="number").columns.tolist()
    transformers = []
    if numeric_features:
        transformers.append(
            ("numeric", SimpleImputer(strategy="median"), numeric_features)
        )
    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )
    if not transformers:
        raise ValueError("At least one usable feature column is required.")

    preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main() -> int:
    try:
        cars = prepare_dataset(load_dataset())
        x_train, x_test, y_train, y_test = split_dataset(cars)
        model = build_model(x_train)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        print(f"Rows used: {len(cars)}")
        print(f"Mean absolute error (MAE): {mean_absolute_error(y_test, predictions):.3f}")
        print(f"Root mean squared error (RMSE): {mean_squared_error(y_test, predictions) ** 0.5:.3f}")
        print(f"R-squared (R2): {r2_score(y_test, predictions):.3f}")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model_path = OUTPUT_DIR / "used_car_price_model.joblib"
        joblib.dump(model, model_path)
        print(f"Saved fitted pipeline to {model_path}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
