# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
import time
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
from decision_tree_lgb import lgb_fit
from prepare_features import add_groupby_features, add_time_features
from sklearn.preprocessing import Imputer, MinMaxScaler

dtypes = {
    'ip'                    : 'uint32',
    'app'                   : 'uint16',
    'device'                : 'uint16',
    'os'                    : 'uint16',
    'channel'               : 'uint16',
    'is_attributed'         : 'uint8',
    'click_id'              : 'uint32'
    }

extra_dtypes = {
        'day'                   : 'uint8',
        'hour'                  : 'uint8',
        'minute'                : 'uint8',
        'second'                : 'uint8',
        'doy'                   : 'uint8',
        'wday'                  : 'uint8',
        'ratio_ips_by_users'    : 'float',
        'clicks_per_ip'         : 'uint8',
        'app_popularity'        : 'uint8',
        'ip_app_ch_h'           : 'uint8',
        'ip_day_hour_c'         : 'uint8',
        'ip_app_os_ch_c'        : 'uint8'
}

train_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_columns  = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']

training_path = "data/train_balanced_sample_4.csv"
validation_path = "data/valid_balanced_sample_4.csv"
testing_path = "data/test.csv"

print('loading train data...')
train_df = pd.read_csv(training_path, dtype=dtypes, usecols=train_columns)
train_df['click_time'] = pd.to_datetime(train_df['click_time'])

print('loading valid data...')
valid_df = pd.read_csv(validation_path, dtype=dtypes, usecols=train_columns)
valid_df['click_time'] = pd.to_datetime(valid_df['click_time'])

print('loading test data...')
test_df = pd.read_csv(testing_path, dtype=dtypes, usecols=test_columns)
test_df['click_time'] = pd.to_datetime(test_df['click_time'])

y_train_df = train_df['is_attributed']
train_df.drop(['is_attributed'], axis=1, inplace=True)

y_valid_df = valid_df['is_attributed']
valid_df.drop(['is_attributed'], axis=1, inplace=True)

click_ids = test_df[['click_id']]
test_df.drop(['click_id'], axis=1, inplace=True) # remover o id de submissão para feature engineering agrupada com treino e validação

nb_train_samples = len(train_df)
nb_valid_samples = len(valid_df)

merge = pd.concat([train_df, valid_df, test_df], sort=False)
merge = add_time_features(merge)
merge = add_groupby_features(merge)
merge = merge.fillna(merge.mean())

x_train_df, x_valid_df, x_test_df = merge.iloc[:nb_train_samples, :], merge.iloc[nb_train_samples:nb_train_samples+nb_valid_samples, :], merge.iloc[nb_train_samples+nb_valid_samples:, :]
x_train_df.index  = train_df.index # corrigir os indices que foram alterados depois do merge
x_valid_df.index  = valid_df.index # corrigir os indices que foram alterados depois do merge
x_test_df.index = test_df.index # corrigir os indices que foram alterados depois do merge

del train_df, valid_df, test_df
gc.collect()

### Preprocessing nan values ###
# # sklearn version
# imputer = Imputer(missing_values='NaN', strategy='mean')
# x_train_df = pd.DataFrame(data=imputer.fit_transform(x_train_df.iloc[:, :]), index=x_train_df.index, columns=x_train_df.columns)
# x_valid_df = pd.DataFrame(data=imputer.fit_transform(x_valid_df.iloc[:, :]), index=x_valid_df.index, columns=x_valid_df.columns)
# x_test_df  = pd.DataFrame(data=imputer.fit_transform(x_test_df.iloc[:, :]),  index=x_test_df.index,  columns=x_test_df.columns)

# pandas version
# x_train_df = x_train_df.fillna(x_train_df.mean())
# x_valid_df = x_valid_df.fillna(x_train_df.mean())
# x_test_df = x_test_df.fillna(x_train_df.mean())

x_test_df = pd.concat([x_test_df, click_ids], axis=1)
x_test_df.to_csv('data/test_preprocessed_sanity_check.csv', index=False)

print("train size: ", len(x_train_df))
print("valid size: ", len(x_valid_df))
print("test size : ", len(x_test_df))

train_df = pd.concat([x_train_df, y_train_df], axis=1)
valid_df = pd.concat([x_valid_df, y_valid_df], axis=1)

del x_test_df # não será utilizado agora...
gc.collect()

train_df.info(verbose=False)

target = 'is_attributed'
all_features = list(train_df.columns)
all_features.remove('is_attributed')

categorical_features = ['ip', 'app', 'device', 'os', 'channel']
datatime_features = ['day', 'hour', 'minute', 'second', 'doy', 'wday']
categorical_features.extend(datatime_features)

gc.collect()

print("Training...")
start_time = time.time()

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric':'auc',
    'learning_rate': 0.05,
    #'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': 48,  # 2^max_depth - 1
    'max_depth': 10,  # -1 means no limit
    # 'min_child_weight' : 0,
    'min_child_samples': 10,
    # 'min_child_samples': 10,  # Minimum number of data need in a child(min_data_in_leaf)
    'subsample': 0.94,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.64,  # Subsample ratio of columns when constructing each tree.
    # "subsample_for_bin": 1000000,
    'min_child_weight': 3,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    # 'scale_pos_weight':200 # because training data is extremely unbalanced 
}

best_model = lgb_fit(lgb_params, 
                        train_df, 
                        valid_df, 
                        all_features, 
                        target, 
                        objective='binary', 
                        metrics='auc',
                        early_stopping_rounds=50, 
                        verbose_eval=True, 
                        num_boost_round=2000, 
                        categorical_features=categorical_features)

print('[{}]: model training time'.format(time.time() - start_time))

del train_df
del valid_df
gc.collect()

# Load the test for predict 
test_df = pd.read_csv('data/test_preprocessed_sanity_check.csv', usecols=all_features + ['click_id'],  dtype=dtypes)

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')

# Save the predictions
print("Predicting...")
sub['is_attributed'] = best_model.predict(test_df[all_features])
print("writing...")
sub.to_csv('sub_lgb_balanced_4_{}.csv'.format(datetime.datetime.now()),index=False)
print("done...")