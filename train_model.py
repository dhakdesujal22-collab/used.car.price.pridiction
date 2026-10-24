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


def load_dataset() -> pd.DataFrame:
    """Load a CSV from data/ only if it contains the expected target column."""
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV dataset found in {DATA_DIR}. Follow data/README.md to obtain "
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
        f"None of the CSV files in {DATA_DIR} contains the required "
        f"`{TARGET}` column. See data/README.md for the expected CarDekho file."
    )


def main() -> int:
    try:
        cars = load_dataset()
        cars = cars.dropna(subset=[TARGET]).copy()
        cars[TARGET] = pd.to_numeric(cars[TARGET], errors="coerce")
        cars = cars.dropna(subset=[TARGET])
        if len(cars) < 2:
            raise ValueError("At least two rows with a valid Selling_Price are required.")

        features = cars.drop(columns=[TARGET])
        target = cars[TARGET]
        numeric_features = features.select_dtypes(include="number").columns.tolist()
        categorical_features = features.select_dtypes(exclude="number").columns.tolist()

        preprocessing = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    SimpleImputer(strategy="median"),
                    numeric_features,
                ),
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_features,
                ),
            ],
            remainder="drop",
        )
        model = Pipeline(
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
        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
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
