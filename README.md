# Used Cars Cost Prediction

First Acmegrade internship project: predict a used car's `Selling_Price` using the public CarDekho dataset.

## Project contents

- `notebooks/used_car_price_prediction.ipynb` — guided data exploration, cleaning, training, evaluation, and conclusions.
- `data/` — put the real CarDekho CSV here; see [data/README.md](data/README.md).
- `train_model.py` — reusable command-line training script.
- `submission/` — saved model and HTML notebook report (created when the notebook is run and exported).

## Python environment

A project-local `.venv` was created with Python 3.11 and the packages in `requirements.txt` were installed. The project VS Code setting points to this interpreter:

```text
.venv\Scripts\python.exe
```

Open this project folder in VS Code, then use **Python: Select Interpreter** and choose the `.venv` entry if VS Code has not selected it already. For notebooks, choose the same `.venv` in the notebook kernel picker.

To recreate the environment later, first install Python 3.10 or newer, then open a terminal in this project folder and run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, you can skip activation and call the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the notebook

1. The public CarDekho CSV should be at `data/car data.csv`; see [data/README.md](data/README.md).
2. In VS Code, open this project folder, select `.venv\Scripts\python.exe` with **Python: Select Interpreter**, then select the same `.venv` kernel in the notebook's kernel picker.
3. Run the notebook cells from top to bottom. Or execute and export it from a terminal in this project folder:

   ```powershell
   .\.venv\Scripts\python.exe -m jupyter nbconvert --to html --execute notebooks/used_car_price_prediction.ipynb --output used_car_price_prediction.html --output-dir .
   ```

The command executes the notebook and writes an HTML report containing the notebook's code, outputs, and charts to the project root. It will stop rather than create a report if execution fails.

## Tests

The isolated unit tests use small synthetic rows solely to verify program behavior; they are not used for model results. Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Train from the command line

```powershell
.\.venv\Scripts\python.exe train_model.py
```

The script reports MAE, RMSE, and R-squared on the same reproducible held-out 20% test split as the notebook, and saves the fitted pipeline to `submission/used_car_price_model.joblib`. It stops with an error if the dataset or required packages are missing. The model's preprocessing is fitted only on the training split.

## Important

Evaluation results depend on the exact dataset and the random train/test split. Do not report any accuracy until the notebook or script has run successfully on the real CarDekho data.
