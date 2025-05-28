import os
import pytest
import pandas as pd
from unittest.mock import patch, mock_open
from cpm.datasets import load_csv, load_bandit_data, load_risky_choices


@pytest.fixture
def mock_csv_data():
    """Fixture to provide mock CSV data."""
    return "col1,col2\n1,2\n3,4"


@patch("os.path.exists")
@patch("pandas.read_csv")
def test_load_csv(mock_read_csv, mock_exists, mock_csv_data):
    # Mock the file existence check
    mock_exists.return_value = True

    # Mock pandas.read_csv to return a DataFrame
    mock_read_csv.return_value = pd.DataFrame({"col1": [1, 3], "col2": [2, 4]})

    # Call the function
    result = load_csv("test.csv")

    # Assertions
    # Get the path that should be used by load_csv
    import cpm.datasets.base
    base_dir = os.path.dirname(cpm.datasets.base.__file__)
    data_dir = os.path.join(base_dir, "data")
    expected_path = os.path.join(data_dir, "test.csv")
    
    mock_exists.assert_called_once_with(expected_path)
    mock_read_csv.assert_called_once()
    assert isinstance(result, pd.DataFrame)
    assert result.equals(pd.DataFrame({"col1": [1, 3], "col2": [2, 4]}))


@patch("os.path.exists")
def test_load_csv_file_not_found(mock_exists):
    # Mock the file existence check to return False
    mock_exists.return_value = False

    # Call the function and assert it raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="No such file or directory:"):
        load_csv("nonexistent.csv")


@patch("cpm.datasets.base.load_csv")
def test_load_bandit_data(mock_load_csv):
    # Mock load_csv to return a DataFrame
    mock_load_csv.return_value = pd.DataFrame({"col1": [1, 3], "col2": [2, 4]})

    # Call the function
    result = load_bandit_data()

    # Assertions
    mock_load_csv.assert_called_once_with("bandit_small.csv")
    assert isinstance(result, pd.DataFrame)
    assert result.equals(pd.DataFrame({"col1": [1, 3], "col2": [2, 4]}))


@patch("cpm.datasets.base.load_csv")
def test_load_risky_choices(mock_load_csv):
    # Mock load_csv to return a DataFrame
    mock_load_csv.return_value = pd.DataFrame({"col1": [1, 3], "col2": [2, 4]})

    # Call the function
    result = load_risky_choices()

    # Assertions
    mock_load_csv.assert_called_once_with("risky_choices.csv")
    assert isinstance(result, pd.DataFrame)
    assert result.equals(pd.DataFrame({"col1": [1, 3], "col2": [2, 4]}))