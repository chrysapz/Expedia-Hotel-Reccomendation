import time
import argparse
import pickle
import os
import gc

import pandas
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
#import seaborn as sns
#from sklearn import metrics



def load_data(file_path):
    gc.collect()
    print("Loading data from file {} ...".format(file_path))
    df = pandas.read_csv(file_path)
    print("Finished loading data....")
    return df

def explore_data(df, train=1):
    print("Exectuting df.head() ...")
    print(df.head())

    # Get the summary statistics
    summary_stats = df.describe()

    # Get the number of unique values for each attribute
    unique_values = df.nunique()

    print("Summary stats: \n", summary_stats)
    print("\n")
    print("Unique vals: ", unique_values)
    print("\n")

    print("This is row 1: ", df.iloc[0])

    # Print general information about the dataset
    print("General Information:")
    print(df.info())

    # Print the number of entries (rows)
    print("\nNumber of Entries (rows):", len(df))

    # Print the number of columns
    print("Number of Columns:", len(df.columns))

    # Print the number of missing values per column
    missing_values = df.isnull().sum()
    print("\nMissing Values per Column:")
    print(missing_values)

    # Print the total number of missing values in the dataset
    total_missing_values = missing_values.sum()
    print("\nTotal Missing Values in Dataset:", total_missing_values)

    print("Feature types: ")
    print(df.dtypes)

    if train == 1:
        print("Actual bookings in train: ", df.loc[(df.booking_bool == 1)].shape[0])
        print("Clicks: ", df.loc[(df.click_bool == 1)].shape[0])
        print("Chance of booking: ", df.loc[(df.booking_bool == 1)].shape[0] / df[df.booking_bool.notnull()].shape[0] * 100, '%')
    

    return 


def plot_correlation(df):
    pass


if __name__ == "__main__":
    df_test = load_data("test_set_VU_DM.csv")
    df = load_data("training_set_VU_DM.csv")
    print("Exploration of training set: \n")
    explore_data(df)

    print("#################### \n ################### \n")
    #print("Exploration of test set: \n")
    #explore_data(df_test,train = 0)