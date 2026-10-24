# CarDekho dataset

The project uses the public **Vehicle Dataset from CarDekho**, published on Kaggle:

<https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho>

The original Kaggle dataset download was retrieved from Kaggle's public dataset-download endpoint (dataset version 1). The CSV in this folder is the real downloaded file, not generated or sampled data:

- `car data.csv` — 301 rows and 9 columns.
- Prediction target: `Selling_Price`.
- Expected columns: `Car_Name`, `Year`, `Selling_Price`, `Present_Price`, `Kms_Driven`, `Fuel_Type`, `Seller_Type`, `Transmission`, `Owner`.

The CSV is ignored by Git so the dataset is not accidentally committed or pushed. If you need to obtain it again, open the Kaggle page above, sign in if prompted, download and extract the dataset, then place the file at:

```text
used-car-price-prediction/data/car data.csv
```

The notebook and training script validate the target column and stop with an error if no suitable CSV is present. They do not create replacement data.
