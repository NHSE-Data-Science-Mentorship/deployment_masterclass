import pandas as pd
from sklearn.model_selection import (
    train_test_split as sklearn_train_test_split,
)

CUTOFF_DATE = pd.to_datetime("2021-01-01")
MIN_LAG = 21


def filter_dates(df, cutoff_date=CUTOFF_DATE, min_lag=MIN_LAG):
    df = df.loc[
        df.date.ge(cutoff_date)
        & df.date.lt(df.date.max() - pd.Timedelta(min_lag, "d"))
    ]
    return df


def get_lag_columns(df, num_days=21):
    df = df.copy()
    for i in range(num_days):
        df[f"lag_{i:0=2d}"] = df["newCasesByPublishDate"].shift(i)
    return df


def prepare_data(df):
    df = filter_dates(df.dropna()).sort_values(by=["areaCode", "date"])

    df_lagged = (
        df.groupby("areaCode")[["date", "newCasesByPublishDate"]]
        .apply(get_lag_columns)
        .reset_index()
        .drop(columns=["level_1"])
        .dropna()
        .merge(
            df,
            left_on=["areaCode", "date", "newCasesByPublishDate"],
            right_on=["areaCode", "date", "newCasesByPublishDate"],
        )
        .drop(columns=["newCasesByPublishDate"])
    )

    feature_cols = [c for c in df_lagged.columns if c.startswith("lag_")]
    target_col = "newCasesBySpecimenDate"
    return (
        df_lagged[["areaCode", "areaName", "date", *feature_cols, target_col]],
        feature_cols,
        target_col,
    )


def train_test_split(df_features, feature_cols, target_col):

    X_train, X_test, y_train, y_test = sklearn_train_test_split(
        df_features[feature_cols], df_features[target_col]
    )
    return X_train, X_test, y_train, y_test
