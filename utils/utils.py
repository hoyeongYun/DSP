import pandas as pd
import numpy as np

def load_data(path: str):
    data_total_df = pd.read_csv(path)
    return data_total_df

def get_transformed_store_id(preprocs: dict, raw_store_id: int):
    return preprocs['MinMaxScaler']['Store'].transform(raw_store_id)