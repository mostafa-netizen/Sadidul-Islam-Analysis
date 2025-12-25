import pandas as pd


if __name__ == '__main__':
    words = pd.read_csv('data/final.csv')
    print(words.head())
    df = words.loc[words.value.str.contains("TENDONS").fillna(False)]
    print(df.head())
    print(df.shape)
