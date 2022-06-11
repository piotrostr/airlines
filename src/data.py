import pandas as pd

from sklearn.model_selection import train_test_split


def get_data() -> pd.DataFrame:
    """
    Returns a dataframe of the data
    """
    df = pd.read_csv('data/data.csv', low_memory=False)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    return df


def get_data_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = get_data()
    train_df, test_df = train_test_split(df, test_size=0.05)
    return train_df, test_df


def get_XY(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = df[['hhi', 'mshare_rc', 'lcc']]
    Y = df['price']
    return X, Y
