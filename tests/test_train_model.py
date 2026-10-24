"""Isolated software tests use synthetic rows; they are not project results."""

from pathlib import Path
import math
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_model import build_model, load_dataset, prepare_dataset, split_dataset


def synthetic_cars(row_count: int = 10) -> pd.DataFrame:
    """Return small synthetic rows for unit tests only."""
    return pd.DataFrame(
        {
            "Year": list(range(2010, 2010 + row_count)),
            "Kms_Driven": [10_000 * index for index in range(row_count)],
            "Fuel_Type": ["Petrol" if index % 2 else "Diesel" for index in range(row_count)],
            "Selling_Price": [float(index + 1) for index in range(row_count)],
        }
    )


class TrainModelTests(unittest.TestCase):
    def test_load_dataset_selects_csv_with_selling_price(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            pd.DataFrame({"not_target": [1]}).to_csv(data_dir / "a.csv", index=False)
            expected = synthetic_cars()
            expected.to_csv(data_dir / "b.csv", index=False)

            loaded = load_dataset(data_dir)

        self.assertEqual(list(loaded.columns), list(expected.columns))
        self.assertEqual(len(loaded), len(expected))

    def test_load_dataset_reports_missing_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(FileNotFoundError, "No CSV dataset found"):
                load_dataset(Path(temp_dir))

    def test_prepare_dataset_removes_duplicates_and_invalid_targets(self) -> None:
        cars = synthetic_cars(3)
        cars["Selling_Price"] = cars["Selling_Price"].astype(object)
        cars.loc[2, "Selling_Price"] = "not a number"
        cars = pd.concat([cars, cars.iloc[[0]]], ignore_index=True)

        prepared = prepare_dataset(cars)

        self.assertEqual(len(prepared), 2)
        self.assertTrue(pd.api.types.is_numeric_dtype(prepared["Selling_Price"]))
        self.assertFalse(prepared["Selling_Price"].isna().any())

    def test_split_and_pipeline_train_and_predict_with_missing_values(self) -> None:
        cars = synthetic_cars()
        cars.loc[0, "Kms_Driven"] = float("nan")
        prepared = prepare_dataset(cars)
        x_train, x_test, y_train, y_test = split_dataset(prepared)

        self.assertEqual(len(x_test), 2)
        self.assertTrue(set(x_train.index).isdisjoint(x_test.index))

        model = build_model(x_train)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        self.assertEqual(len(predictions), len(y_test))
        self.assertTrue(all(math.isfinite(float(value)) for value in predictions))


if __name__ == "__main__":
    unittest.main()
