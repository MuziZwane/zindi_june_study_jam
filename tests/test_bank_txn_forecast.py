from pathlib import Path
import py_compile

import pytest


def test_script_compiles():
    py_compile.compile("src/bank_txn_forecast.py", doraise=True)


def test_pipeline_writes_log_space_submission(tmp_path: Path, monkeypatch):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("sklearn")

    from src.bank_txn_forecast import TARGET_COLUMN, main

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    train = pd.DataFrame(
        {
            "UniqueID": [f"customer_{idx}" for idx in range(30)],
            "segment": ["mass" if idx % 2 else "affluent" for idx in range(30)],
            "age": [25 + idx for idx in range(30)],
            TARGET_COLUMN: [(idx % 7) + 1 for idx in range(30)],
        }
    )
    test = pd.DataFrame(
        {
            "UniqueID": ["customer_30", "customer_31", "customer_32"],
            "segment": ["mass", "affluent", "mass"],
            "age": [35, 44, 52],
        }
    )
    transactions = pd.DataFrame(
        {
            "UniqueID": [f"customer_{idx % 33}" for idx in range(120)],
            "txn_date": pd.date_range("2015-01-01", periods=120, freq="D"),
            "amount": [10.0 + (idx % 13) for idx in range(120)],
            "channel": ["atm" if idx % 3 else "app" for idx in range(120)],
        }
    )

    train.to_csv(data_dir / "Train.csv", index=False)
    test.to_csv(data_dir / "Test.csv", index=False)
    transactions.to_csv(data_dir / "transactions.csv", index=False)

    output = tmp_path / "submission.csv"
    monkeypatch.setattr(
        "sys.argv",
        ["bank_txn_forecast.py", "--data-dir", str(data_dir), "--output", str(output)],
    )

    main()

    submission = pd.read_csv(output)
    assert submission.columns.tolist() == ["UniqueID", TARGET_COLUMN]
    assert len(submission) == 3
    assert submission[TARGET_COLUMN].ge(0).all()
