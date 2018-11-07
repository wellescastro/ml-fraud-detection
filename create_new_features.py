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

training_path = "data/train_small.pkl.gz"
validation_path = "data/valid_small.pkl.gz"

train_df = pd.read_pickle(training_path)
train_df['click_time'] = pd.to_datetime(train_df['click_time'])

valid_df = pd.read_pickle(validation_path)
valid_df['click_time'] = pd.to_datetime(valid_df['click_time'])

nb_train_samples = len(train_df)
train_valid_df = pd.concat([train_df, valid_df])


########## Adding time features #############

# Extract different time granulatities

train_valid_df['day'] = train_valid_df['click_time'].dt.day.astype('uint8')
train_valid_df['hour'] = train_valid_df['click_time'].dt.hour.astype('uint8')
train_valid_df['minute'] = train_valid_df['click_time'].dt.minute.astype('uint8')
train_valid_df['second'] = train_valid_df['click_time'].dt.second.astype('uint8')
train_valid_df["doy"]  =  train_valid_df['click_time'].dt.dayofyear.astype('uint8')
train_valid_df['wday'] =  train_valid_df['click_time'].dt.dayofweek.astype('uint8')


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
# device_count = train_valid_df.groupby(['device'])['is_attributed'].count() # count how many clicks came from the same device
# downloaded_device_count = train_valid_df.groupby(['device'])['is_attributed'].sum() # take only the positive ones, the ones that were downloaded
# rate = downloaded_device_count / device_count
# rate = rate.to_frame().reset_index().rename(index=str, columns={'is_attributed':'device_downloadability'})
# train_extra = train_valid_df.merge(rate, on='device', how='left')

variables = ['ip', 'app', 'device', 'os', 'channel']
for var in variables:
    train_valid_df = add_frequency_features(train_valid_df, [var])

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
    train_valid_df = add_groupby_feature(train_valid_df, groupby_criteria, features_to_select, function, feature_name)

print ("Final features")
print(train_valid_df.columns)

train_df, valid_df = train_valid_df.iloc[:nb_train_samples, :], train_valid_df.iloc[nb_train_samples:, :]

print("Training data shape: ", train_df.shape)
print("Validation data shape: ", valid_df.shape)

train_df.to_pickle('data/train_preprocessed.pkl.gz')
valid_df.to_pickle('data/valid_preprocessed.pkl.gz')