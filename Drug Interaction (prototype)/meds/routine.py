import os
import pandas as pd
import glob2

def main():

    list_of_jpgs = glob2.glob("/raid/data2/asanogo/meds/object_detector/*/*/*.jpg")
    df = pd.DataFrame(list_of_jpgs)
    new_df = df.sample(frac=1)

    training_set = new_df.iloc[:int(0.8*new_df.shape[0])]
    test_set = new_df.iloc[int(0.8 * new_df.shape[0]):]

    training_set.to_csv("/raid/data2/asanogo/meds/training.csv", index=None, header=False)
    test_set.to_csv("/raid/data2/asanogo/meds/test.csv", index=None, header=False)
    print("the training set has {0} samples".format(training_set.shape[0]))
    print("the test set has {0} samples".format(test_set.shape[0]))
if __name__=='__main__':
    main()