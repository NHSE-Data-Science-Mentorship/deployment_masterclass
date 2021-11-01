from case_regression_example.cleaning import train_test_split, get_lag_columns
import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def fake_df():
    return pd.DataFrame(
        data={
            "x": np.linspace(100, 200, 100),
            "y": np.random.randint(0, 100, 100),
            "z": np.random.normal(50, 5, 100),
        }
    )


def test_train_test_split_features(fake_df):
    X_train, _, __, ___ = train_test_split(fake_df, ["z", "x"], "y")
    assert isinstance(X_train, pd.DataFrame)
    assert list(X_train.columns) == ["z", "x"]


def test_train_test_split_features(fake_df):
    _, __, ___, y_test = train_test_split(fake_df, ["z", "x"], "y")
    assert isinstance(y_test, pd.Series)
    assert y_test.name == "y"


def test_train_test_split_shapes(fake_df):
    X_train, y_train, X_test, y_test = train_test_split(
        fake_df, ["z", "x"], "y"
    )

    assert X_train.shape[0] == y_train.shape[0]  # Same number of rows
    assert X_train.shape[1] == 2  # We have 2 features
    assert y_train.shape[1] == 1

    assert X_test.shape[0] == y_train.shape[0]  # Same number of rows
