import pandas as pd
import pytest

from src.load_data import load_csv, DataLoadError


def test_load_csv_success(tmp_path):
    csv_path = tmp_path / "temp_test.csv"
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    df.to_csv(csv_path, index=False)

    result = load_csv(csv_path)

    assert result.source.endswith("temp_test.csv")
    assert result.df.shape == (3, 2)
    assert list(result.df.columns) == ["A", "B"]
    assert result.df.iloc[0]["A"] == 1
    assert result.df.iloc[0]["B"] == "x"


def test_load_csv_missing_file(tmp_path):
    missing_path = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        load_csv(missing_path)


def test_load_csv_wrong_extension(tmp_path):
    bad_path = tmp_path / "not_a_csv.txt"
    bad_path.write_text("hello", encoding="utf-8")

    with pytest.raises(DataLoadError):
        load_csv(bad_path)


def test_load_csv_empty_csv(tmp_path):
    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("", encoding="utf-8")  # empty file

    with pytest.raises(DataLoadError):
        load_csv(empty_path)

def test_load_csv_nrows(tmp_path):
    csv_path = tmp_path / "big.csv"
    df = pd.DataFrame({"A": range(100), "B": range(100)})
    df.to_csv(csv_path, index=False)

    result = load_csv(csv_path, nrows=20)

    assert result.df.shape == (20, 2)


def test_load_dataset_nrows(tmp_path):
    csv_path = tmp_path / "train.csv"
    df = pd.DataFrame({"A": range(50), "B": range(50)})
    df.to_csv(csv_path, index=False)

    from src.load_data import load_dataset
    result = load_dataset(csv_path, nrows=10)

    assert result["train"].df.shape[0] == 10
