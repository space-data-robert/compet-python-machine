import gc
import pandas as pd

def makeParquet(fname):
    dframe = pd.read_csv(
        f'{fname}.csv'
    )
    dframe.to_parquet(
        f'{fname}.parquet'
    )

    del dframe
    gc.collect()