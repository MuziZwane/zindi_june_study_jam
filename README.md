# Zindi June Study Jam: Bank Transaction Volume Forecasting

This repository contains a reproducible baseline for the **Bank Transaction Volume Forecasting Challenge**. The goal is to predict each customer's total number of transactions over November 2015 through January 2016.

## Key challenge detail

Zindi expects the submission target to be log transformed. The generated submission therefore writes:

```python
np.log1p(predicted_transaction_count)
```

for the `next_3m_txn_count` column, rather than raw transaction counts.

## Project layout

```text
src/bank_txn_forecast.py  # End-to-end feature engineering, training, and submission script
tests/                   # Lightweight regression tests with synthetic data
requirements.txt         # Python dependencies
```

## Expected data layout

Download the competition files from Zindi and place the CSV files in a local folder, for example `data/`:

```text
data/
  Train.csv
  Test.csv
  transactions.csv       # optional: any extra customer-level or event-level CSVs
  monthly_snapshots.csv  # optional
  demographics.csv       # optional
```

The script automatically discovers additional CSV files in the data folder, joins them by `UniqueID`, and aggregates repeated customer records into modelling features.

## Run the baseline

```bash
python src/bank_txn_forecast.py --data-dir data --output submission.csv
```

The script will:

1. Load `Train.csv`, `Test.csv`, and any other CSV files in `--data-dir`.
2. Build customer-level aggregate features from numeric, categorical, and date-like columns.
3. Train a preprocessing + `HistGradientBoostingRegressor` model on `log1p(next_3m_txn_count)`.
4. Save a two-column submission with `UniqueID` and log-transformed `next_3m_txn_count` predictions.

## Validation

When the training set is large enough, the script prints a 5-fold out-of-fold RMSE on the log-transformed target. This matches the platform's scoring setup because the competition evaluates RMSE on log values.
