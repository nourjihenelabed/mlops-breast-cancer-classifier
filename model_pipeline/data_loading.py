import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Dataset is empty")
    return df
