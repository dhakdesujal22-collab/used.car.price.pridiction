# Obtain the real CarDekho dataset

This project needs the real, public CarDekho used-car dataset. No dataset was found in the workspace, and no data has been fabricated.

1. Open the Kaggle dataset page: <https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho>.
2. Sign in to Kaggle if prompted and download the dataset ZIP using the page's **Download** button.
3. Extract the archive. Use the CSV named `car data.csv` (the version with the exact column headers `Car_Name`, `Year`, `Selling_Price`, `Present_Price`, `Kms_Driven`, `Fuel_Type`, `Seller_Type`, `Transmission`, and `Owner`).
4. Copy that CSV into this project's `data` folder and name it exactly:

   ```text
   car data.csv
   ```

The finished location should be `used-car-price-prediction/data/car data.csv`. The CSV must contain a numeric `Selling_Price` column. Do not substitute an unrelated dataset or create sample values; the project intentionally stops if the real data is missing.
