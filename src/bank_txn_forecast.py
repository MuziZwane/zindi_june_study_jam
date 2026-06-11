"""Train a baseline model for the Zindi bank transaction forecasting challenge.

The competition submission requires log1p-transformed predictions. This script trains
on log1p(target) and writes those log-space predictions directly to the submission.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ID_COLUMN = "UniqueID"
TARGET_COLUMN = "next_3m_txn_count"
DATE_KEYWORDS = ("date", "month", "period", "time")
RECENCY_WINDOWS = (3, 6, 12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing Train.csv, Test.csv, and optional extra CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission.csv"),
        help="Path where the Zindi submission file will be written.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=2026,
        help="Random seed used for validation folds and model training.",
    )
    return parser.parse_args()


def load_csv(data_dir: Path, filename: str) -> pd.DataFrame:
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Expected {path} to exist")
    return pd.read_csv(path)


def discover_extra_tables(data_dir: Path) -> list[Path]:
    reserved = {"train.csv", "test.csv", "sample_submission.csv", "samplesubmission.csv"}
    return sorted(path for path in data_dir.glob("*.csv") if path.name.lower() not in reserved)


def find_date_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [col for col in frame.columns if any(key in col.lower() for key in DATE_KEYWORDS)]
    date_columns: list[str] = []
    for col in candidates:
        parsed = pd.to_datetime(frame[col], errors="coerce")
        if parsed.notna().mean() >= 0.5:
            date_columns.append(col)
    return date_columns


def add_datetime_parts(frame: pd.DataFrame, table_name: str, date_columns: Iterable[str]) -> pd.DataFrame:
    enriched = frame.copy()
    for col in date_columns:
        parsed = pd.to_datetime(enriched[col], errors="coerce")
        prefix = f"{table_name}_{col}"
        enriched[f"{prefix}_year"] = parsed.dt.year
        enriched[f"{prefix}_month"] = parsed.dt.month
        enriched[f"{prefix}_quarter"] = parsed.dt.quarter
        enriched[f"{prefix}_dayofweek"] = parsed.dt.dayofweek
    return enriched


def aggregate_recent_windows(
    frame: pd.DataFrame,
    id_column: str,
    date_column: str,
    numeric_columns: list[str],
    table_name: str,
) -> pd.DataFrame:
    dated = frame[[id_column, date_column, *numeric_columns]].copy()
    dated[date_column] = pd.to_datetime(dated[date_column], errors="coerce")
    dated = dated.dropna(subset=[date_column])
    if dated.empty:
        return pd.DataFrame({id_column: frame[id_column].drop_duplicates()})

    max_date = dated[date_column].max()
    parts = []
    row_counts = dated.groupby(id_column).size().rename(f"{table_name}_dated_row_count")
    parts.append(row_counts)

    for months in RECENCY_WINDOWS:
        cutoff = max_date - pd.DateOffset(months=months)
        recent = dated[dated[date_column] > cutoff]
        count = recent.groupby(id_column).size().rename(f"{table_name}_last{months}m_row_count")
        parts.append(count)
        if numeric_columns and not recent.empty:
            recent_numeric = recent.groupby(id_column)[numeric_columns].agg(["sum", "mean", "max"])
            recent_numeric.columns = [
                f"{table_name}_last{months}m_{column}_{stat}" for column, stat in recent_numeric.columns
            ]
            parts.append(recent_numeric)

    features = pd.concat(parts, axis=1).reset_index()
    return features


def aggregate_table(path: Path, id_column: str = ID_COLUMN) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if id_column not in frame.columns:
        return pd.DataFrame(columns=[id_column])

    table_name = path.stem.lower().replace(" ", "_").replace("-", "_")
    date_columns = find_date_columns(frame)
    enriched = add_datetime_parts(frame, table_name, date_columns)

    excluded = {id_column, TARGET_COLUMN, *date_columns}
    numeric_columns = [col for col in enriched.select_dtypes(include=[np.number]).columns if col not in excluded]
    categorical_columns = [
        col
        for col in enriched.select_dtypes(include=["object", "category", "bool"]).columns
        if col not in excluded and col != id_column
    ]

    grouped = enriched.groupby(id_column, dropna=False)
    aggregates: list[pd.DataFrame] = []

    record_count = grouped.size().rename(f"{table_name}_record_count").to_frame()
    aggregates.append(record_count)

    if numeric_columns:
        numeric_agg = grouped[numeric_columns].agg(["mean", "std", "min", "max", "sum", "last"])
        numeric_agg.columns = [f"{table_name}_{column}_{stat}" for column, stat in numeric_agg.columns]
        aggregates.append(numeric_agg)

    for col in categorical_columns:
        nunique = grouped[col].nunique(dropna=True).rename(f"{table_name}_{col}_nunique").to_frame()
        non_null = grouped[col].count().rename(f"{table_name}_{col}_non_null_count").to_frame()
        aggregates.extend([nunique, non_null])

    features = pd.concat(aggregates, axis=1).reset_index()

    for date_column in date_columns[:1]:
        recent = aggregate_recent_windows(frame, id_column, date_column, numeric_columns, table_name)
        features = features.merge(recent, on=id_column, how="left")

    return features


def merge_extra_features(base: pd.DataFrame, extra_tables: Iterable[Path]) -> pd.DataFrame:
    merged = base.copy()
    for path in extra_tables:
        features = aggregate_table(path)
        if not features.empty and ID_COLUMN in features.columns:
            merged = merged.merge(features, on=ID_COLUMN, how="left")
    return merged


def build_model(random_state: int) -> Pipeline:
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, make_column_selector("number")),
            ("categorical", categorical_transformer, make_column_selector("category")),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    regressor = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=400,
        l2_regularization=0.05,
        random_state=random_state,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", regressor)])


def make_column_selector(kind: str):
    def selector(frame: pd.DataFrame) -> list[str]:
        if kind == "number":
            return frame.select_dtypes(include=[np.number]).columns.tolist()
        if kind == "category":
            return frame.select_dtypes(exclude=[np.number]).columns.tolist()
        raise ValueError(f"Unsupported selector kind: {kind}")

    return selector


def prepare_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.drop(columns=[TARGET_COLUMN], errors="ignore").copy()
    if ID_COLUMN in features.columns:
        features = features.drop(columns=[ID_COLUMN])
    for col in features.select_dtypes(include=["object", "category", "bool"]).columns:
        features[col] = features[col].astype("string").fillna("missing")
    return features


def validate_model(model: Pipeline, x_train: pd.DataFrame, y_log: pd.Series, random_state: int) -> None:
    if len(x_train) < 25:
        print("Skipping cross-validation because fewer than 25 training rows are available.")
        return
    splits = min(5, len(x_train))
    cv = KFold(n_splits=splits, shuffle=True, random_state=random_state)
    oof = cross_val_predict(model, x_train, y_log, cv=cv, n_jobs=None)
    rmse = float(np.sqrt(mean_squared_error(y_log, oof)))
    print(f"{splits}-fold log-space RMSE: {rmse:.5f}")


def write_submission(test_ids: pd.Series, predictions_log: np.ndarray, output: Path) -> None:
    submission = pd.DataFrame(
        {
            ID_COLUMN: test_ids,
            TARGET_COLUMN: np.clip(predictions_log, a_min=0.0, a_max=None),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"Wrote {len(submission):,} rows to {output}")


def main() -> None:
    args = parse_args()
    train = load_csv(args.data_dir, "Train.csv")
    test = load_csv(args.data_dir, "Test.csv")

    missing_columns = {ID_COLUMN, TARGET_COLUMN} - set(train.columns)
    if missing_columns:
        raise ValueError(f"Train.csv is missing required columns: {sorted(missing_columns)}")
    if ID_COLUMN not in test.columns:
        raise ValueError(f"Test.csv is missing required column: {ID_COLUMN}")

    extra_tables = discover_extra_tables(args.data_dir)
    print(f"Discovered {len(extra_tables)} extra CSV table(s): {[path.name for path in extra_tables]}")

    train_features = merge_extra_features(train, extra_tables)
    test_features = merge_extra_features(test, extra_tables)

    x_train = prepare_matrix(train_features)
    x_test = prepare_matrix(test_features)
    x_test = x_test.reindex(columns=x_train.columns, fill_value=np.nan)
    y_log = np.log1p(train[TARGET_COLUMN].clip(lower=0))

    model = build_model(args.random_state)
    validate_model(model, x_train, y_log, args.random_state)
    model.fit(x_train, y_log)
    predictions_log = model.predict(x_test)
    write_submission(test[ID_COLUMN], predictions_log, args.output)


if __name__ == "__main__":
    main()
