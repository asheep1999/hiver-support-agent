import pandas as pd

DATA_PATH = "data/raw/twcs.csv"

def load_data(nrows=None):
    df = pd.read_csv(DATA_PATH, nrows=nrows)
    return df

if __name__=="__main__":
    df = load_data(nrows=10000)

    print("Shape:", df.shape)

    print("\n Columns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

