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

training_path = "data/X_train_small.pkl.gz"

X_train = pd.read_pickle(training_path)
X_train['click_time'] = pd.to_datetime(X_train['click_time'])

# X_test = pd.read_csv("data/test.csv")
# X_test['click_time'] = pd.to_datetime(X_test['click_time'])

############# Organizing the features ###############

categorical_variables = ['ip', 'app', 'device', 'os', 'channel', 'is_attributed']
for v in categorical_variables:
    X_train[v] = X_train[v].astype('category')
    
    if v is not 'is_attributed': # this variable occurs only in the training set
        # X_test[v] = X_test[v].astype('category')
        pass


############# Show basic info ###############

print("Training data information")
print(X_train.info())
print("\n\n")
print("Basic statistical measures")
print(X_train.describe())
print("\n\n")

## visual description of the unique counts

feature_names = categorical_variables[:-1] # :-1 to exclude the output variable and shows only the input features
# get unique counts of each feature
uniques = [len(X_train[col].unique()) for col in feature_names] 
# configure the plot
plt.figure(figsize=(10, 6))
sns.set(font_scale=1.2)
# create the barplot
ax = sns.barplot(feature_names, uniques, log=True) # set log to visualize better the counts across different variables with different magnitudes
ax.set(xlabel='Features', ylabel='Unique Counts', title='Number of unique values per feature (N samples: {})'.format(len(X_train)))
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    index = p.get_x() + p.get_width() / 2. # 0, 1, 2, 3, 4
    ax.text(index,
            height + 10,
            uniq,
            ha="center") 
plt.show()


## visual description of the output variable

ax_single = sns.countplot(x='is_attributed', data=X_train, log=True)
uniques = list(X_train['is_attributed'].value_counts())
for p, uniq in zip(ax_single.patches, uniques):
    height = p.get_height()
    ax_single.text(p.get_x()+p.get_width()/2.,
            height+0.01,
            uniq,
            ha="center")

plt.show()