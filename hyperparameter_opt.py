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
import lightgbm as lgb
from prepare_features import add_groupby_features, add_time_features
from sklearn.preprocessing import Imputer, MinMaxScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from hyperopt import hp, tpe
from hyperopt.fmin import fmin


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

training_path = "data/train_balanced_sanity_check.csv"
validation_path = "data/valid_balanced_sanity_check.csv"
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

# train_df = add_time_features(train_df)
# train_df = add_groupby_features(train_df)

y_valid_df = valid_df['is_attributed']
valid_df.drop(['is_attributed'], axis=1, inplace=True)

# valid_df = add_time_features(valid_df)
# valid_df = add_groupby_features(valid_df)

click_ids = test_df[['click_id']]
test_df.drop(['click_id'], axis=1, inplace=True) # remover o id de submissão para feature engineering agrupada com treino e validação

# test_df = add_time_features(test_df)
# test_df = add_groupby_features(test_df)

# print("test size : ", len(click_ids))
# exit()
nb_train_samples = len(train_df)
nb_valid_samples = len(valid_df)

merge = pd.concat([train_df, valid_df, test_df], sort=False)
merge = add_time_features(merge)
# merge = add_groupby_features(merge)

del train_df, valid_df, test_df
gc.collect()

x_train_df, x_valid_df, x_test_df = merge.iloc[:nb_train_samples, :], merge.iloc[nb_train_samples:nb_train_samples+nb_valid_samples, :], merge.iloc[nb_train_samples+nb_valid_samples:, :]
x_test_df.index = click_ids.index # corrigir os indices que foram alterados depois do merge
x_test_df = pd.concat([x_test_df, click_ids], axis=1)

### Preprocessing nan values ###
# # sklearn version
# imputer = Imputer(missing_values='NaN', strategy='mean')
# x_train_df = pd.DataFrame(data=imputer.fit_transform(x_train_df.iloc[:, :]), index=x_train_df.index, columns=x_train_df.columns)
# x_valid_df = pd.DataFrame(data=imputer.fit_transform(x_valid_df.iloc[:, :]), index=x_valid_df.index, columns=x_valid_df.columns)
# x_test_df  = pd.DataFrame(data=imputer.fit_transform(x_test_df.iloc[:, :]),  index=x_test_df.index,  columns=x_test_df.columns)

# pandas version
x_train_df = x_train_df.fillna(x_train_df.mean())
x_valid_df = x_valid_df.fillna(x_valid_df.mean())
x_test_df = x_test_df.fillna(x_test_df.mean())

x_test_df.to_csv('data/test_preprocessed_sanity_check.csv', index=False)

print("train size: ", len(x_train_df))
print("valid size: ", len(x_valid_df))
print("test size : ", len(x_test_df))

train_df = pd.concat([x_train_df, y_train_df], axis=1)
valid_df = pd.concat([x_valid_df, y_valid_df], axis=1)

del x_test_df # não será utilizado agora...
gc.collect()

# train_df.info(verbose=False)

target = 'is_attributed'
all_features = list(x_train_df.columns)
# all_features.remove('is_attributed')

categorical_features = ['ip', 'app', 'device', 'os', 'channel']
datatime_features = ['day', 'hour', 'minute', 'second', 'doy', 'wday']
categorical_features.extend(datatime_features)

gc.collect()

print("Starts tuning LightGBM...")
start_time = time.time()


train_loader = lgb.Dataset(train_df[all_features].values, label=train_df[target].values,
                    feature_name=all_features,
                    # categorical_feature=categorical_features,
                    free_raw_data=False
                    )
valid_loader = lgb.Dataset(valid_df[all_features].values, label=valid_df[target].values,
                        feature_name=all_features,
                        # categorical_feature=categorical_features,
                        free_raw_data=False
                        )

fixed_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric':'auc',
    'learning_rate': 0.05,    
    'verbose': -1,
    'nthread': 4,
}

opt_params = dict(**fixed_params)

def objective(params):

    opt_params.update(
        {    
        'num_leaves': (2 ** int(params['max_depth'])) - 1,
        'max_depth': int(params['max_depth']),
        'min_child_samples': int(params['min_child_samples']),
        'min_child_weight': '{:.3f}'.format(params['min_child_weight']),  # means something like "stop trying to split once your sample size in a node goes below a given threshold"
        'feature_fraction': '{:.3f}'.format(params['feature_fraction']),
        # 'bagging_fraction': '{:.3f}'.format(params['bagging_fraction']),
        # 'bagging_freq': int(params['bagging_freq']),
        # 'lambda_l1':float(params['lambda_l1']),
        # 'lambda_l2':float(params['lambda_l2']),
        }
    )
    
    # clf = lgb.LGBMClassifier(
    #     n_estimators=500,
    #     learning_rate=0.01,
    #     **params
    # )
    # fit_params = {
    #     'valid_sets':[train_loader, valid_loader], 
    #     'valid_names':['train','valid'],
    #     'early_stopping_rounds':30
    # }
    # score = cross_val_score(clf, x_train_df, y_train_df, scoring='roc_auc', cv=StratifiedKFold(), fit_params=fit_params).mean()

    results = {}

    bst1 = lgb.train(opt_params, 
                     train_loader, 
                     valid_sets=[train_loader, valid_loader], 
                     valid_names=['train','valid'], 
                     evals_result=results, 
                     num_boost_round=500,
                     early_stopping_rounds=30,
                     verbose_eval=0,
                     categorical_feature=categorical_features
                     )

    n_estimators = bst1.best_iteration
    train_score = results['train']['auc'][n_estimators-1]
    valid_score = results['valid']['auc'][n_estimators-1]

    print("train_auc: {:.4f} valid_auc: {:.4f} params {}".format(train_score, valid_score, opt_params))
    return 1 - valid_score

space = {
    'num_leaves': hp.choice('num_leaves', np.arange(15, 255, 5, dtype=int)),
    'max_depth': hp.choice('max_depth', [4, 6, 8, 10]),
    'min_child_samples': hp.choice('min_child_samples', np.arange(10, 30, 5, dtype=int)),
    'min_child_weight' : hp.uniform('min_child_weight', 1, 10), # both min_child_samples and min_child_weight means something like "stop trying to split once your sample size in a node goes below a given threshold"
    'feature_fraction': hp.uniform('feature_fraction', 0.5, 1.0),
    # 'lambda_l1':hp.choice('lambda_l1', [0, 0.5, 1.0]),
    # 'lambda_l2':hp.choice('lambda_l2', [0, 0.5, 1.0]),
    # 'bagging_fraction': hp.uniform('bagging_fraction', 0.8, 1.0),
    # 'bagging_freq' : hp.choice('bagging_freq', [0,1]),
}

# best_optimized = fmin(fn=objective,
#             space=space,
#             algo=tpe.suggest,
#             max_evals=10)
# best_params = dict(fixed_params, **best_optimized)

best_params = {'verbose': -1, 'feature_fraction': 0.7339856029443689, 'bagging_fraction': 0.8990249943846076, 'learning_rate': 0.1, 'nthread': 4, 'min_child_weight': 3.9499190833540627, 'max_depth': 2, 'objective': 'binary', 'min_child_samples': 3, 'bagging_freq': 0, 'metric': 'auc', 'boosting_type': 'gbdt'}
with open('best_params_{}.log'.format(datetime.datetime.now()), 'w') as f:
    f.write(str(best_params))

# del train_loader
# del valid_loader
gc.collect()

full_train = pd.concat([train_df, valid_df], axis=0)

train_loader = lgb.Dataset(full_train[all_features].values, label=full_train[target].values,
                    feature_name=all_features,
                    # categorical_feature=categorical_features,
                    free_raw_data=False
                    )


final_model = lgb.train(best_params, 
                    train_loader, 
                    valid_sets=[train_loader],
                    num_boost_round=5000,
                    early_stopping_rounds=5,
                    verbose_eval=10,
                    categorical_feature=categorical_features
                )
del train_df
del valid_df
del full_train
gc.collect()

# Load the test for predict 
test_df = pd.read_csv('data/test_preprocessed_sanity_check.csv', usecols=all_features + ['click_id'],  dtype=dtypes)

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')

# Save the predictions
print("Predicting...")
sub['is_attributed'] = final_model.predict(test_df[all_features])
print("writing...")
sub.to_csv('sub_lgb_balanced_{}.csv'.format(datetime.datetime.now()),index=False)
print("done...")