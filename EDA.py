import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

'''
Exploratory data analysis (EDA)

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

############# Read Data ###############

training_path = "data/train_small.pkl.gz"

train = pd.read_pickle(training_path)
train['click_time'] = pd.to_datetime(train['click_time'])

# test = pd.read_csv("data/test.csv")
# test['click_time'] = pd.to_datetime(test['click_time'])

############# Organizing the features ###############

categorical_variables = ['ip', 'app', 'device', 'os', 'channel', 'is_attributed']
for v in categorical_variables:
    train[v] = train[v].astype('category')
    
    if v is not 'is_attributed': # this variable occurs only in the training set
        # test[v] = test[v].astype('category')
        pass


############# Show basic info ###############

print("Training data information")
print(train.info())
print("\n\n")
print("Basic statistical measures")
print(train.describe())
print("\n\n")

## visual description of the unique counts

feature_names = categorical_variables[:-1] # :-1 to exclude the output variable and shows only the input features
# get unique counts of each feature
uniques = [len(train[col].unique()) for col in feature_names] 
# configure the plot
plt.figure(figsize=(10, 6))
sns.set(font_scale=1.2)
# create the barplot
ax = sns.barplot(feature_names, uniques, log=True) # set log to visualize better the counts across different variables with different magnitudes
ax.set(xlabel='Features', ylabel='Unique Counts', title='Number of unique values per feature (N samples: {})'.format(len(train)))
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    index = p.get_x() + p.get_width() / 2. # 0, 1, 2, 3, 4
    ax.text(index,
            height + 10,
            uniq,
            ha="center") 
plt.show()


## visual description of the output variable

asingle = sns.countplot(x='is_attributed', data=train, log=True)
uniques = list(train['is_attributed'].value_counts())
for p, uniq in zip(asingle.patches, uniques):
    height = p.get_height()
    asingle.text(p.get_x()+p.get_width()/2.,
            height+0.01,
            uniq,
            ha="center")

plt.show()