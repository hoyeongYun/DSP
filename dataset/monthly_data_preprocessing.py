import pandas as pd
import numpy as np
from torch.utils.data import  Dataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def create_monthly_prediction_df(data_total_df: pd.DataFrame):
    """
    process data required at monthly prediction task
    key: column name
    value: preprocessor correspond to each column
    {
        'LabelEncoder':{
            'Month' : le_Month,
            'Urban_Rural' : le_Urban_Rural
        },
        'MinMaxScaler':{
            'Store' : mms_Store         # added after data split 
            'Store_Owner' : mms_Store_Owner
            'Latitude' : mms_Latitude
            'Longitude' : mms_Longitude
            'Month_Industry_Size' : mms_Month_Industry_Size
            'Month_Retail_Size' : mms_Month_Retail_Size
        }
    }
    """
    monthly_prediction_columns = ['Month', 'Store', 'Store_Owner', 'Urban_Rural', 'Latitude', 'Longitude', 'Month_Industry_Size', 'Month_Retail_Size']
    temp_df = data_total_df[monthly_prediction_columns]
    temp_df = temp_df.groupby(['Store', 'Month'])[monthly_prediction_columns].min().reset_index(drop=True)

    categorical = ['Month', 'Urban_Rural']
    numerical = ['Store', 'Store_Owner', 'Latitude', 'Longitude', 'Month_Industry_Size', 'Month_Retail_Size']
    preprocs = {'LabelEncoder':{},
                'MinMaxScaler':{}}

    for col_name in monthly_prediction_columns:
        if col_name in categorical:
            preprocs['LabelEncoder'][col_name] = LabelEncoder()
            cur_encoder = preprocs['LabelEncoder'][col_name]
            temp_df[col_name] = cur_encoder.fit_transform(temp_df[col_name])
        else:
            preprocs['MinMaxScaler'][col_name] = MinMaxScaler()
            cur_scaler = preprocs['MinMaxScaler'][col_name]
            temp_df[col_name] = cur_scaler.fit_transform(temp_df[col_name].to_numpy().reshape(-1,1))

    # create test_df
    test_df = temp_df[temp_df['Month'] != 0].reset_index(drop=True)        # 642*86, 7
    # create data_monthly_df ( train + eval )
    target = temp_df[temp_df['Month'] != 0].Month_Retail_Size.reset_index(drop=True)
    data_monthly_df = temp_df[temp_df['Month'] != 86].reset_index(drop=True)
    data_monthly_df['Target'] = target
        
    return data_monthly_df, test_df, preprocs

def split_monthly_prediction_df(df: pd.DataFrame, strategy: int, preprocs: dict):
    '''
    split ratio:    578 : 64 == 9 : 1
    split strategy: 1. 위도 경도로 clustering 한 뒤, cluster 크기 비율로 추출해서 64개 맞추기
                    2. store별 monthly retail size가 0인 비율 얻고, 구간을 나누어 하나씩 추출해 64개 맞추기
                    3. 1, 2 번 혼합
    '''
    if strategy == 1:
        result = split_strategy_1(df, preprocs)
    elif strategy == 3:
        result = split_strategy_3(df, preprocs)
    else:
        result = split_strategy_2(df, preprocs)
    train_df, eval_df = result
    return train_df, eval_df

# default split strategy
def split_strategy_2(data_monthly_df: pd.DataFrame, preprocs: dict, seed: int = 42, num_stores: int = 642, classes: int = 8, test_set_ratio: float = 0.1):
    '''
    Args:
        df: total data frame
        preprocs: transform instances applied to each columns
    '''
    test_set_ratio = 0.1
    NUM_STORES = num_stores
    NUM_STORES_TEST = int(NUM_STORES * test_set_ratio)        # 9:1 split  -> 578 : 64
    DIVIDED_CLASSES = classes                      # 8 classes로 분할 후 각 class마다 0.1 fraction씩 sampling
    np.random.seed(seed)

    mme_store = preprocs['MinMaxScaler']['Store']

    # 각 store의 zero 값이 몇개인지 담을 list
    zero_datas = []
    for store_id in range(1, NUM_STORES+1):
        cur_store_id = mme_store.transform(np.array([[store_id]])).item()   # EX) 91 -> 0.00XXX
        cur_store_df = data_monthly_df[data_monthly_df['Store'] == cur_store_id]
        num_zeros = len(cur_store_df[cur_store_df['Month_Retail_Size'] == 0.0])
        zero_datas.append(num_zeros)

    temp = pd.cut(pd.Series(zero_datas), DIVIDED_CLASSES, labels=[i for i in range(DIVIDED_CLASSES)]) # 0인 데이터가 많을 수록 label value 증가
    temp.index = np.arange(1, NUM_STORES+1)

    eval_store_ids = np.array([[]])
    left = NUM_STORES_TEST
    for i in range(DIVIDED_CLASSES):
        if i == 7:
            store_ids = temp[temp == i].index
            eval_store_id = np.random.choice(store_ids, left, replace=False).reshape(1, -1)
            eval_store_ids = np.concatenate((eval_store_ids, eval_store_id), axis=1)
        else:
            store_ids = temp[temp == i].index
            samples = int(len(store_ids)*test_set_ratio)
            left -= samples
            eval_store_id = np.random.choice(store_ids, samples, replace=False).reshape(1, -1)
            eval_store_ids = np.concatenate((eval_store_ids, eval_store_id), axis=1)

    eval_store_ids = np.sort(eval_store_ids.flatten())                  # [64, ]
    total_store_ids = np.arange(1, NUM_STORES+1)
    train_store_ids = np.setdiff1d(total_store_ids, eval_store_ids)     # [578, ]

    assert len(eval_store_ids) == len(np.unique(eval_store_ids))
    assert len(eval_store_ids) + len(train_store_ids) == len(total_store_ids)
    
    eval_df = pd.DataFrame()
    for eval_store_id in eval_store_ids:
        cur_store_id = mme_store.transform(np.array([[eval_store_id]])).item()
        temp = data_monthly_df[data_monthly_df['Store'] == cur_store_id]
        eval_df = pd.concat([eval_df, temp])

    train_df = pd.DataFrame()
    for train_store_id in train_store_ids:
        cur_store_id = mme_store.transform(np.array([[train_store_id]])).item()
        temp = data_monthly_df[data_monthly_df['Store'] == cur_store_id]
        train_df = pd.concat([train_df, temp])

    result = [train_df, eval_df]
    return result

def split_strategy_1(df: pd.DataFrame, preprocs: dict):
    pass

def split_strategy_3(df: pd.DataFrame, preprocs: dict):
    pass