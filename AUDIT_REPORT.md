# Used Cars Cost Prediction - Validation Report

## Results

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Preserve and validate the existing project and notebook | PASS | Updated the existing notebook in place; retained its 16 cells and validated the notebook JSON. |
| 2 | Use a legitimate public CarDekho dataset | PASS | Downloaded dataset version 1 from the public Kaggle CarDekho dataset endpoint. Source and re-download instructions are in `data/README.md`. |
| 3 | Verify target, columns, types, missing values, and path | PASS | `data/car data.csv` has 301 rows, 9 expected columns, and numeric `Selling_Price`. No cells are missing values; 2 duplicate rows are removed during preparation. |
| 4 | Avoid data leakage and use an appropriate split | PASS | A reproducible shuffled 80/20 split is made before fitting. Imputation and one-hot encoding are inside a pipeline fitted only on training rows. |
| 5 | Keep notebook and training script consistent | PASS | Both use the shared loading, cleaning, splitting, and model-building functions in `train_model.py`. |
| 6 | Install dependencies and support VS Code Python | PASS | Python 3.11.17 and project requirements are installed in the local `.venv`. Project VS Code settings point at that interpreter. For interactive notebook use, select this same environment in the notebook kernel picker if VS Code has not already done so. |
| 7 | Execute all notebook cells and calculate metrics | PASS | All 7 code cells executed in order with no error outputs. Notebook and training script agree on the results below. |
| 8 | Export HTML from the executed notebook | PASS | The report exists in the project root and includes code, outputs, and embedded charts. |
| 9 | Add and run meaningful tests | PASS | Four isolated `unittest` cases passed, covering CSV selection and missing data, target cleaning, train/test splitting, imputation, and prediction. Synthetic rows are used only inside these software tests. |
| 10 | Produce this audit and list remaining actions | PASS | No dataset, model metrics, or report content was fabricated. |

## Dataset and model results

- Dataset: 301 source rows and 9 columns; `Selling_Price` is the prediction target.
- Data types: `Year`, `Kms_Driven`, and `Owner` are integers; `Selling_Price` and `Present_Price` are floating-point; `Car_Name`, `Fuel_Type`, `Seller_Type`, and `Transmission` are text.
- Missing values: 0 in every column.
- Duplicate rows: 2 removed; 299 rows remained.
- Split: 239 training rows and 60 held-out test rows.
- MAE: **1.365** dataset price units.
- RMSE: **3.310** dataset price units.
- R-squared: **0.575**.

These are results from this dataset and fixed random split, not a guarantee of future performance. The notebook export emitted an accessibility warning that three chart images lack alternative text; the charts are present in the report.

## Verified files

- Notebook: `notebooks/used_car_price_prediction.ipynb`
- HTML report: `used_car_price_prediction.html`
- Dataset: `data/car data.csv` (excluded from Git by `.gitignore`)
- Training script: `train_model.py`
- Dependencies: `requirements.txt`
- Dataset source and instructions: `data/README.md`
- Unit tests: `tests/test_train_model.py`
- VS Code interpreter setting: `.vscode/settings.json`

## Remaining manual step

For interactive work in VS Code, open the `used-car-price-prediction` folder, choose `.venv\Scripts\python.exe` with **Python: Select Interpreter**, and choose the same `.venv` in the notebook kernel picker if needed. The notebook has already been executed and the HTML report generated using this project environment.
