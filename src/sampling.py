import pandas as pd


def top_sample_df(df: pd.DataFrame, n_rows: int, cols=["value"]) -> pd.DataFrame:
    if len(df) < n_rows:
        df = _oversample_df(df, n_rows)
    else:
        df = df.nlargest(n_rows, columns=cols)
    return df


def random_sample_df(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    if len(df) < n_rows:
        df = _oversample_df(df, n_rows)
    else:
        # take random subset without replacement
        df = df.sample(n_rows, replace=False)
    return df


def _oversample_df(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    # take all rows at least once plus random subset with replacement
    return pd.concat(
        [
            df,
            df.sample(n_rows - len(df), replace=True),
        ]
    )


def limit_sample_df(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    if len(df) > n_rows:
        # take random subset without replacement
        df = df.sample(n_rows, replace=False)
    return df
