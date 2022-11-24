import pandas as pd
import numpy as np
from torch.utils.data import  Dataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from utils.utils import *

# COL_NAMES = ['Month', 'Store', 'Store_Owner', 'Urban_Rural', 'Latitude', 
#             'Longitude', 'Item_Type', 'Item', 'Industry_Size', 'Retail_Size', 'Target_3_Month_Retail_Sum']

def get_model_keys(df: pd.DataFrame):
    '''
    데이터상 zero 인 비율이 너무 크기 때문에, sparse한 정도에 따라 model_type을 0, 1, 2 로 나누고,
    0, 1 인 데이터에 대해서만 각각 모델 적용
    df: End_To_End_Data.csv
    '''

    def num_of_zeros(retail):
        if retail == 0:
            return 1
        else:
            return 0

    temp_df = df[['Store', 'Month', 'Item', 'Retail_Size']].copy()
    temp_df['num_of_zeros'] = temp_df['Retail_Size'].apply(num_of_zeros)

    store_item_df = temp_df.groupby(['Store', 'Item'])[['num_of_zeros']].sum()
    store_item_df = store_item_df.reset_index() # groupby해서 순서바뀜 그냥 이렇게 가져오지 말고 원본 데이터 column 그대로 복사해서 가져오기
    store_item_df['Item'] = pd.Categorical(store_item_df['Item'], ['Power Cord', 'Phone Charger', 'Ear Buds', 'Mouse', 'Keyboard', 'Milk', 'Eggs', 'Cereal', 'Shrimp', 'Noodles', 'Steak', 'King Crab', 'Tape', 'Glue', 'Nails', 'Bracket', 'Brush', 'Paint'])
    store_item_df = store_item_df.sort_values(['Store', 'Item'])
    store_item_df = store_item_df.reset_index(drop=True)
    store_item_df = store_item_df.reset_index()
    store_item_df = store_item_df.rename(columns={'index' : 'Item_Store_Key'})

    threshold = [0, 40, 70, 87]
    model_type = pd.cut(store_item_df.num_of_zeros, threshold, labels=[0, 1, 2])
    store_item_df['Model_Type'] = pd.Series(model_type)

    print('# of model 0 data : ', len(store_item_df[store_item_df.Model_Type == 0]))
    print('# of model 1 data : ', len(store_item_df[store_item_df.Model_Type == 1]))
    print('# of model 2 data : ', len(store_item_df[store_item_df.Model_Type == 2]))

    model_0_keys = store_item_df[store_item_df.Model_Type == 0]['Item_Store_Key'].to_list()
    model_1_keys = store_item_df[store_item_df.Model_Type == 1]['Item_Store_Key'].to_list()
    model_2_keys = store_item_df[store_item_df.Model_Type == 2]['Item_Store_Key'].to_list()

    return (model_0_keys, model_1_keys, model_2_keys)

def get_df_by_model_type(df: pd.DataFrame, model_keys: list):
    '''
    model_key 전달받고, endtoenddata.csv에서 각 model에 넣을 df로 분할
    '''
    def model_type_gen(key):
        if key in model_0_keys:
            return 0
        elif key in model_1_keys:
            return 1
        elif key in model_2_keys:
            return 2

    model_0_keys, model_1_keys, model_2_keys = model_keys
    temp_df = df.copy()
    temp_df['Model_Type'] = temp_df['Item_Store_Key'].apply(model_type_gen)
    df_type_0 = temp_df[temp_df.Model_Type == 0]
    df_type_1 = temp_df[temp_df.Model_Type == 1]

    # assert len(df_type_0)/87 == 351.0
    # assert len(df_type_1)/87 == 1870.0

    return (df_type_0, df_type_1)

def create_per_item_prediction_df(df: pd.DataFrame):
    '''
    '''
    target_col = ['Target_3_Month_Retail_Sum']
    req_cols = ['Month', 'Store', 'Store_Owner', 'Urban_Rural', 'Location_Cluster', 'Item_Type', 
                'Item', 'Industry_Size', 'Retail_Size', 'Target_3_Month_Retail_Sum']
    temp_df = df.copy()
    temp_df = temp_df.sort_values(by=['Item_Store_Key', 'Month']).reset_index(drop=True)[req_cols]
    # preprocs
    categorical_cols = ['Month', 'Urban_Rural']
    numerical_cols = ['Store_Owner', 'Location_Cluster', 'Industry_Size', 
                    'Retail_Size', 'Target_3_Month_Retail_Sum']
    dummy_cols = ['Item_Type', 'Item']
    preprocs = {'LabelEncoder':{}, 
                'MinMaxScaler':{}}
    for col_name in categorical_cols + numerical_cols:
        if col_name in categorical_cols:
            preprocs['LabelEncoder'][col_name] = LabelEncoder()
            cur_encoder = preprocs['LabelEncoder'][col_name]
            temp_df[col_name] = preprocs['LabelEncoder'][col_name].fit_transform(temp_df[col_name])
        else:
            preprocs['MinMaxScaler'][col_name] = MinMaxScaler()
            cur_scaler = preprocs['MinMaxScaler'][col_name]
            temp_df[col_name] = cur_scaler.fit_transform(temp_df[col_name].to_numpy().reshape(-1,1))
    temp_df = pd.get_dummies(temp_df, columns=dummy_cols)
    test_df = temp_df[(temp_df['Month'] != 0) & (temp_df['Month'] != 1) & (temp_df['Month'] != 2)].copy().reset_index(drop=True)
    train_eval_df = temp_df[(temp_df['Month'] != 84) & (temp_df['Month'] != 85) & (temp_df['Month'] != 86)].copy().reset_index(drop=True)

    return train_eval_df, test_df, preprocs


# def train_eval_df_split(df: pd.DataFrame, strategy: int, preprocs: dict):
#     '''
#     split ratio:    578 : 64 == 9 : 1
#     split strategy: 1. 위도 경도로 clustering 한 뒤, cluster 크기 비율로 추출해서 64개 맞추기
#                     2. store별 monthly retail size가 0인 비율 얻고, 구간을 나누어 하나씩 추출해 64개 맞추기
#                     3. 1, 2 번 혼합
#     '''
#     if strategy == 1:
#         result = split_strategy_1(df, preprocs)
#     elif strategy == 3:
#         result = split_strategy_3(df, preprocs)
#     else:
#         result = split_strategy_2(df, preprocs)
#     train_df, eval_df = result
#     return train_df, eval_df

# # default split strategy
# def split_strategy_2(df: pd.DataFrame, preprocs: dict, seed: int = 42, num_stores: int = 642, classes: int = 8, eval_set_ratio: float = 0.1):
#     '''
#     Args:
#         df: total data frame
#         preprocs: transform instances applied to each columns

#     end to end case에서는 store id 가 transformed 되어있지 않음
#     '''
#     eval_set_ratio = 0.1
#     NUM_STORES = num_stores
#     NUM_STORES_EVAL = int(NUM_STORES * eval_set_ratio)        # 9:1 split  -> 578 : 64
#     DIVIDED_CLASSES = classes                      # 8 classes로 분할 후 각 class마다 0.1 fraction씩 sampling
#     np.random.seed(seed)

#     # 각 store의 zero 값이 몇개인지 담을 list
#     zero_datas = []
#     for store_id in range(1, NUM_STORES+1):
#         cur_store_df = df[df['Store'] == store_id]
#         num_zeros = len(cur_store_df[cur_store_df['Target_3_Month_Retail_Sum'] == 0.0])
#         zero_datas.append(num_zeros)

#     temp = pd.cut(pd.Series(zero_datas), DIVIDED_CLASSES, labels=[i for i in range(DIVIDED_CLASSES)]) # 0인 데이터가 많을 수록 label value 증가
#     temp.index = np.arange(1, NUM_STORES+1)

#     eval_store_ids = np.array([[]])
#     left = NUM_STORES_EVAL
#     for i in range(DIVIDED_CLASSES):
#         if i == 7:
#             store_ids = temp[temp == i].index
#             eval_store_id = np.random.choice(store_ids, left, replace=False).reshape(1, -1)
#             eval_store_ids = np.concatenate((eval_store_ids, eval_store_id), axis=1)
#         else:
#             store_ids = temp[temp == i].index
#             samples = int(len(store_ids)*eval_set_ratio)
#             left -= samples
#             eval_store_id = np.random.choice(store_ids, samples, replace=False).reshape(1, -1)
#             eval_store_ids = np.concatenate((eval_store_ids, eval_store_id), axis=1)

#     eval_store_ids = np.sort(eval_store_ids.flatten())                  # [64, ]
#     total_store_ids = np.arange(1, NUM_STORES+1)
#     train_store_ids = np.setdiff1d(total_store_ids, eval_store_ids)     # [578, ]

#     assert len(eval_store_ids) == len(np.unique(eval_store_ids))
#     assert len(eval_store_ids) + len(train_store_ids) == len(total_store_ids)
    
#     eval_df = pd.DataFrame()
#     for eval_store_id in eval_store_ids:
#         temp = df[df['Store'] == eval_store_id]
#         eval_df = pd.concat([eval_df, temp])

#     train_df = pd.DataFrame()
#     for train_store_id in train_store_ids:
#         temp = df[df['Store'] == train_store_id]
#         train_df = pd.concat([train_df, temp])

#     result = [train_df.reset_index(drop=True), eval_df.reset_index(drop=True)]
#     return result

# def split_strategy_1(df: pd.DataFrame, preprocs: dict):
#     pass

# def split_strategy_3(df: pd.DataFrame, preprocs: dict):
#     pass