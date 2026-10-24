# Used Cars Cost Prediction

First Acmegrade internship project: predict a used car's `Selling_Price` using the public CarDekho dataset.

## Project contents

- `notebooks/used_car_price_prediction.ipynb` — guided data exploration, cleaning, training, evaluation, and conclusions.
- `data/` — put the real CarDekho CSV here; see [data/README.md](data/README.md).
- `train_model.py` — reusable command-line training script.
- `submission/` — saved model and HTML notebook report (created when the notebook is run and exported).

## Before you begin

Python was not available in the inspected workspace when this project was created. Install Python 3.10 or newer and ensure the `py` launcher is available, then open PowerShell in this project folder.

Create and activate a project-local virtual environment:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install this project's dependencies inside that environment:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

These commands have not been run. No packages were installed.

## Run the notebook

1. Obtain and place the real dataset as described in [data/README.md](data/README.md).
2. In the activated environment, start Jupyter:

   ```powershell
   jupyter notebook
   ```

3. Open `notebooks/used_car_price_prediction.ipynb` and run all cells from top to bottom.
4. Export the executed notebook as HTML:

   ```powershell
   jupyter nbconvert --to html --execute notebooks/used_car_price_prediction.ipynb --output used_car_price_prediction.html --output-dir submission
   ```

The HTML report is produced only after the notebook executes successfully with the real dataset.

## Train from the command line

With the dataset at `data/car data.csv`, run:

```powershell
python train_model.py
```

The script reports MAE, RMSE, and R-squared on a held-out test set, and saves the fitted pipeline to `submission/used_car_price_model.joblib`. It stops with a clear error if the dataset or required packages are missing. The script does not use fabricated data.

## Important

Evaluation results depend on the exact dataset and the random train/test split. Do not report any accuracy until the notebook or script has run successfully on the real CarDekho data.
