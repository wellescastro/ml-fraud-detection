import pandas as pd
import numpy as np
'''
Data info
columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        }
'''

training_path = "data/X_train_small.pkl.gz"
validation_path = "data/X_valid_small.pkl.gz"

X_train = pd.read_pickle(training_path)
X_train['click_time'] = pd.to_datetime(X_train['click_time'])

X_valid = pd.read_pickle(validation_path)
X_valid['click_time'] = pd.to_datetime(X_valid['click_time'])

nb_train_samples = len(X_train)
X_train_valid = pd.concat([X_train, X_valid])


########## Adding time features #############

# Extract different time granulatities

X_train_valid['day'] = X_train_valid['click_time'].dt.day.astype('uint8')
X_train_valid['hour'] = X_train_valid['click_time'].dt.hour.astype('uint8')
X_train_valid['minute'] = X_train_valid['click_time'].dt.minute.astype('uint8')
X_train_valid['second'] = X_train_valid['click_time'].dt.second.astype('uint8')
X_train_valid["doy"]  =  X_train_valid['click_time'].dt.dayofyear.astype('uint8')
X_train_valid['wday'] =  X_train_valid['click_time'].dt.dayofweek.astype('uint8')


########## Adding frequency features - approach based on groupby->count() ##############

# generalized function to add frequency features
def add_frequency_features(df, features):
    groupby_obj = df.groupby(features)['is_attributed']
    rate = groupby_obj.sum() / groupby_obj.count()
    # scaling step 
    scales = []
    for i in groupby_obj.count():
        scales.append(min([1, np.log(i) / np.log(100000)]))
    rate = rate * scales
    rate = rate.to_frame().reset_index().rename(index=str, columns={'is_attributed': '{}_downloadability'.format(''.join(features))})
    return df.merge(rate, on=features, how='left')

# Add device downloadability feature: frequência referente a quantas vezes os clicks oriundo de cada um dos dispositivo resultaram no download do aplicativo
# device_count = X_train_valid.groupby(['device'])['is_attributed'].count() # count how many clicks came from the same device
# downloaded_device_count = X_train_valid.groupby(['device'])['is_attributed'].sum() # take only the positive ones, the ones that were downloaded
# rate = downloaded_device_count / device_count
# rate = rate.to_frame().reset_index().rename(index=str, columns={'is_attributed':'device_downloadability'})
# train_extra = X_train_valid.merge(rate, on='device', how='left')

variables = ['ip', 'app', 'device', 'os', 'channel']
for var in variables:
    X_train_valid = add_frequency_features(X_train_valid, [var])

######### Adding others features by grouping #############
''' 
Pandas grouping functionality
df.groupby(by=grouping_columns)[columns_to_show].function()

function -> count, size, mean, var
'''

def add_groupby_feature(df, groupby_crit, cols_to_select, function, feature_name):
    groupby_obj = df.groupby(groupby_crit)[cols_to_select].agg(function)
    groupby_obj = groupby_obj.reset_index().rename(index=str, columns={cols_to_select[0]: feature_name})
    return df.merge(groupby_obj, on=groupby_crit, how='left')


ratio_ips_by_users = lambda x: float(len(x)) / len(x.unique())

grouping_strategies = {
    ratio_ips_by_users :[['app'], ['ip'], 'ratio_ips_by_users'],  # for each app, given the list of ip's associated with the app, we count the total number of ips and divide it by the number of unique ips (distinct users) [se eu fizer apenas a contagem direta dos ips associados, pode ser que uma contagem alta de ips seja em razao de um unico usuario clicar milhares de vezes em um app]
    'count': [['app'], ['channel'], 'app_popularity'], # How popular is the app ?
    # other extra combinations
    'count': [['ip','app', 'channel'], ['hour'], 'ip_app_ch_h'], # group by ip-app-channel
    'count': [['ip','day', 'hour'], ['channel'], 'ip_day_hour_c'], # group by ip-day-hour
    'var': [['ip', 'app', 'os'], ['hour'], 'ip_app_os_hour_cnt'], # group by ip-app-os
    'count': [['ip', 'app', 'os'], ['channel'], 'ip_app_os_ch_var'] # group by ip-app-os
}

for function, groupby in grouping_strategies.items():
    groupby_criteria = groupby[0]
    features_to_select = groupby[1]
    feature_name = groupby[2]
    X_train_valid = add_groupby_feature(X_train_valid, groupby_criteria, features_to_select, function, feature_name)

print ("Final features")
print(X_train.columns)

X_train, X_valid = X_train_valid.iloc[:nb_train_samples, :], X_train_valid.iloc[nb_train_samples:, :]

print(X_train.shape)
print(X_valid.shape)
X_train.to_pickle('data/X_train_preprocessed.pkl.gz')
X_valid.to_pickle('data/X_valid_preprocessed.pkl.gz')