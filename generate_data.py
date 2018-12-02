import pandas as pd
import numpy as np
import gc
import random


#-------------- Read and extract training and validation data ---------------#

train_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_columns  = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
 #      'click_id'      : 'uint32'
        }

# # Testing data
# print( "Extracting testing data...")
# test = pd.read_csv( "data/test.csv", 
#                       usecols=test_columns, # the last column is not presented in the testing set 
#                       dtype=dtypes)

# test.to_pickle('data/test.csv')
# gc.collect()
# exit()
generate_small_dataset = False
generate_balanced = True

assert((generate_balanced is False and generate_small_dataset is False) or (generate_balanced != generate_small_dataset) )

#------- Generate smaller training and validation data for faster sanity check of the remaining codes -----------#

if generate_small_dataset is True:
        # Training data
        print( "Extracting training data...")
        training = pd.read_csv( "data/train.csv", 
                                nrows=1000000,  # original value -> 122071523
                                usecols=train_columns, 
                                dtype=dtypes)

        print("\nTraining data:")
        print(training.shape )
        print(training.head() )
        print("Saving training data...")
        training.to_pickle('train_small.pkl.gz')

        gc.collect()

        # Validation data
        print( "Extracting first chunk of validation data...")
        valid1 = pd.read_csv( "data/train.csv", 
                        skiprows=range(1,1000000), 
                        nrows=100000, 
                        usecols=train_columns, 
                        dtype=dtypes)

        valid1.reset_index(drop=True,inplace=True)
        valid1.to_pickle('data/valid_small.pkl.gz')

        gc.collect()

elif generate_balanced is True:
    validation_holdout = 0.1
    nb_datasets = 1
    for i in range(nb_datasets+1):
        # Training data
        chunksize = 10000 # 100000000
        iter_csv = pd.read_csv('data/train.csv', iterator=True, chunksize=chunksize, usecols=train_columns, dtype=dtypes)
        # df = pd.concat([chunk[chunk['field'] > 1] for chunk in iter_csv])
        final_positives = pd.DataFrame()
        final_negatives = pd.DataFrame()
        
        for idx, chunk in enumerate(iter_csv):
            chunk.reset_index(drop=True, inplace=True)
            positives_indexis = chunk[chunk['is_attributed'] == 1].index.values
            negatives_indexis = chunk[chunk['is_attributed'] == 0].index.values

            if len(positives_indexis) == 0:
                    continue

            assert len(negatives_indexis) > len(positives_indexis)

            positives = chunk.ix[positives_indexis]

            negative_indexis_filtered = np.random.choice(range(len(positives)), len(positives))
            negatives = chunk.ix[negative_indexis_filtered]

            final_positives = pd.concat([final_positives, positives])
            final_negatives = pd.concat([final_negatives, negatives])

            assert len(positives) == len(negatives)

            
        
        final_positives = final_positives.reindex()
        final_negatives = final_negatives.reindex()

        print(final_positives)
        exit()
        validation_positives = random.sample(range(0,len(training)), int(len(training)*validation_holdout*0.5))
        validation_negatives = random.sample(range(0,len(training)), int(len(training)*validation_holdout*0.5))
        validation = pd.concat([final_positives.ix[validation_positives], final_negatives.ix[validation_negatives]], axis=0)

        final_positives.drop(final_positives.index[validation_positives])
        final_negatives.drop(final_negatives.index[validation_negatives])

        training = pd.concat([final_positives, final_negatives], axis=0)
        training = training.sort_values('click_time')
        training.reset_index(drop=True, inplace=True)


        validation_indexis = random.sample(range(0,len(training)),int(len(training)*validation_holdout))
        validation = training.ix[validation_indexis]
        validation.reset_index(drop=True, inplace=True)

        training = training.drop(training.index[validation_indexis])
        
        print("\nTraining data:")
        print(training.shape )
        print(training.head() )
        print("Saving training data...")
        training.to_csv('data/train_balanced_sanity_check_{}.csv'.format(i), index=False)

        print("\nValidation data:")
        print(validation.shape )
        print(validation.head() )
        print("Saving validation data...")
        validation.to_csv('data/valid_balanced_sanity_check_{}.csv'.format(i), index=False)

else:
        #------- Generate real training and validation data samples  -----------#

        # Training data
        print( "Extracting training data...")
        training = pd.read_csv( "data/train.csv", 
                                nrows=122071523,
                                usecols=train_columns, 
                                dtype=dtypes)

        print("\nTraining data:")
        print(training.shape )
        print(training.head() )
        print("Saving training data...")
        training.to_pickle('data/train.pkl.gz')

        gc.collect()

        # Validation data
        print( "Extracting first chunk of validation data...")
        valid1 = pd.read_csv( "data/train.csv", 
                        skiprows=range(1,144708153), 
                        nrows=7705357, 
                        usecols=train_columns, 
                        dtype=dtypes)
        print( "Extracting second chunk of validation data...")
        valid2 = pd.read_csv( "data/train.csv", 
                        skiprows=range(1,161974466), 
                        nrows=6291379, 
                        usecols=train_columns, 
                        dtype=dtypes)
        valid2 = pd.concat([valid1, valid2])
        del valid1
        gc.collect()
        print( "Extracting third chunk of validation data...")
        valid3 = pd.read_csv( "data/train.csv", 
                        skiprows=range(1,174976527), 
                        nrows=6901686, 
                        usecols=train_columns, 
                        dtype=dtypes)
        valid3 = pd.concat([valid2,valid3])
        del valid2
        gc.collect()
        validation = valid3
        del valid3
        gc.collect()

        validation.reset_index(drop=True,inplace=True)
        print("\nValidation data:")
        print(validation.shape)
        print(validation.head())
        print("Saving validation data...")
        validation.to_pickle('data/valid.pkl.gz')

print("\nDone")