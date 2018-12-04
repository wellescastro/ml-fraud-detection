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
from hyperopt import hp, tpe, space_eval
from hyperopt.fmin import fmin
from hyperopt import Trials
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

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

training_path = "data/train_balanced_sample_1.csv"
validation_path = "data/valid_balanced_sample_1.csv"
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

print("train size: ", len(x_train_df))
print("valid size: ", len(x_valid_df))
print("test size : ", len(x_test_df))

del train_df, valid_df, test_df
gc.collect()

# x_test_df = pd.concat([x_test_df, click_ids], axis=1)
# x_test_df.to_csv('data/test_preprocessed_sanity_check.csv', index=False)

train_df = pd.concat([x_train_df, y_train_df], axis=1)
valid_df = pd.concat([x_valid_df, y_valid_df], axis=1)

del x_test_df # o teste não será utilizado agora, só depois da otimização dos parâmetros...

gc.collect()

target = 'is_attributed'
all_features = list(x_train_df.columns)

categorical_features = ['ip', 'app', 'device', 'os', 'channel']
datatime_features = ['day', 'hour', 'minute', 'second']
categorical_features.extend(datatime_features)

gc.collect()

train_df.info()

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


###################### HYPERPARAMETER OPTIMIZATION USING ONE OF THE RANDOMLY SUBSAMPLED SETS ######################

fixed_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric':'auc',
    'learning_rate': 0.05,    
    'verbose': -1,
    'nthread': 4,
    # 'min_child_weight' : 5,
    # 'min_child_samples': 10,
    'bagging_freq' : 1
}

opt_params = dict(**fixed_params)

def objective(params):

    opt_params.update(
        {    
        # 'num_leaves': (2 ** int(params['max_depth'])) - 1,
        'num_leaves': int(params['num_leaves']),
        'max_depth': int(params['max_depth']),
        'feature_fraction': '{:.3f}'.format(params['feature_fraction']),
        'bagging_fraction': '{:.3f}'.format(params['bagging_fraction']),
        'min_child_samples': int(params['min_child_samples']),
        'min_child_weight': '{:.3f}'.format(params['min_child_weight']),
        }
    )
   
    results = {}

    trial = lgb.train(opt_params, 
                     train_loader, 
                     valid_sets=[train_loader, valid_loader], 
                     valid_names=['train','valid'], 
                     evals_result=results, 
                     num_boost_round=500,
                     early_stopping_rounds=30,
                     verbose_eval=0,
                     categorical_feature=categorical_features
                     )

    n_estimators = trial.best_iteration
    train_score = results['train']['auc'][n_estimators-1]
    valid_score = results['valid']['auc'][n_estimators-1]

    print("train_auc: {:.4f} valid_auc: {:.4f} params {}".format(train_score, valid_score, opt_params))
    return 1 - valid_score

space = {
    'num_leaves': hp.choice('num_leaves', np.arange(8, 56, 8, dtype=int)),
    'max_depth': hp.choice('max_depth', [6, 8, 10]),
    'feature_fraction': hp.uniform('feature_fraction', 0.6, 1.0),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.8, 1.0),
    'min_child_samples': hp.choice('min_child_samples', np.arange(10, 25, 5, dtype=int)), #  something like "stop trying to split once your sample size in a node goes below a given threshold"
    'min_child_weight' : hp.uniform('min_child_weight', 2, 5),
}

best_optimized = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=80)

best_params = space_eval(space, best_optimized)
best_params = dict(fixed_params, **best_params)

print("#" * 50)
print("Best Params!\n")
print(best_params)
with open('best_params_{}.log'.format(datetime.datetime.now()), 'w') as f:
    f.write(str(best_params))

############################### ENSEMBLE DIFFERENT GRADIENT BOOSTED TREES WITH TUNED PARAMETERS #################################

del train_df
del valid_df
del train_loader
del valid_loader
gc.collect()

nb_datasets = 5

print('loading test data...')
test_df_reuse = pd.read_csv(testing_path, dtype=dtypes, usecols=test_columns)
test_df_reuse['click_time'] = pd.to_datetime(test_df_reuse['click_time'])
click_ids = test_df_reuse['click_id']

test_df_reuse.drop(['click_id'], axis=1, inplace=True) # remover o id de submissão para feature engineering agrupada com treino e validação

sub_ensemble = pd.DataFrame()
sub_ensemble['click_id'] = click_ids.astype('int')
sub_ensemble['is_attributed'] = 0

for i in range(1, nb_datasets+1):
    train_data = 'data/train_balanced_sample_{}.csv'.format(i)
    valid_data = 'data/valid_balanced_sample_{}.csv'.format(i)

    print('loading train data...')
    train_df = pd.read_csv(train_data, dtype=dtypes, usecols=train_columns)
    train_df['click_time'] = pd.to_datetime(train_df['click_time'])

    y_train_df = train_df['is_attributed']
    train_df.drop(['is_attributed'], axis=1, inplace=True)

    print('loading valid data...')
    valid_df = pd.read_csv(valid_data, dtype=dtypes, usecols=train_columns)
    valid_df['click_time'] = pd.to_datetime(valid_df['click_time'])

    y_valid_df = valid_df['is_attributed']
    valid_df.drop(['is_attributed'], axis=1, inplace=True)

    nb_train_samples = len(train_df)
    nb_valid_samples = len(valid_df)

    merge = pd.concat([train_df, valid_df, test_df_reuse], sort=False)
    merge = add_time_features(merge)
    merge = add_groupby_features(merge)
    merge = merge.fillna(merge.mean())

    x_train_df, x_valid_df, x_test_df = merge.iloc[:nb_train_samples, :], merge.iloc[nb_train_samples:nb_train_samples+nb_valid_samples, :], merge.iloc[nb_train_samples+nb_valid_samples:, :]
    x_train_df.index  = train_df.index # corrigir os indices que foram alterados depois do merge
    x_valid_df.index  = valid_df.index # corrigir os indices que foram alterados depois do merge
    x_test_df.index = test_df_reuse.index # corrigir os indices que foram alterados depois do merge

    del merge
    gc.collect()

    # train_df = pd.concat([x_train_df, y_train_df], axis=1)
    # valid_df = pd.concat([x_valid_df, y_valid_df], axis=1)

    # train_full = pd.concat([train_df, valid_df])

    train_loader = lgb.Dataset(x_train_df[all_features].values, label=y_train_df.values,
                        feature_name=all_features,
                        free_raw_data=False
                    )
    
    valid_loader = lgb.Dataset(x_valid_df[all_features].values, label=y_valid_df.values,
                        feature_name=all_features,
                        free_raw_data=False
                    )

    current_model = lgb.train(best_params, 
                        train_loader, 
                        valid_sets=[train_loader, valid_loader],
                        num_boost_round=500,
                        early_stopping_rounds=30,
                        verbose_eval=100,
                        categorical_feature=categorical_features
                    )

    del train_loader
    del valid_loader
    del train_df
    del valid_df
    gc.collect()

    predictions = current_model.predict(x_test_df[all_features])

    sub_ensemble['is_attributed'] += predictions

    sub_individual = pd.DataFrame()
    sub_individual['click_id'] = click_ids.astype('int')
    sub_individual['is_attributed'] = predictions
    sub_individual.to_csv('sub_lgb_balanced_reducedR1_individual_{}.csv'.format(i),index=False)

    try:
        current_model.save_model('sub_lgb_balanced_reducedR1_individual_{}.txt'.format(i))
        ax = lgb.plot_importance(current_model, max_num_features=100)
        plt.savefig('sub_lgb_balanced_reducedR1_individual_{}.png'.format(i))
    except Exception as e:
        print("não consegui salvar o modelo =/")
    


sub_ensemble['is_attributed'] = sub_ensemble['is_attributed'] / 5. # average predictions

print("writing...")
sub_ensemble.to_csv('sub_lgb_balanced_reducedR1_ensemble_{}.csv'.format(datetime.datetime.now()),index=False)
print("done...")