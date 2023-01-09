import pandas as pd


if __name__ == '__main__':

    data: pd.DataFrame = pd.read_parquet('data/parking.parquet')

    print(f'data.shape = {data.shape}')
