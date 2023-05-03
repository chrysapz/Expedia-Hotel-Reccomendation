import time
import argparse
import pickle
import os
import gc

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
#import seaborn as sns
#from sklearn import metrics


def remap(df):
    col_map = {name: num for num, name in enumerate(df.columns)}
    df = df.rename(columns=col_map)
    return df

def load_data(file_path):
    gc.collect()
    print("Loading data from file {} ...".format(file_path))
    df = pd.read_csv(file_path)
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

    total_missing_values = missing_values.sum()
    print("\nTotal Missing Values in Dataset:", total_missing_values)

    print("Feature types: ")
    print(df.dtypes)

    if train == 1:
        print("Actual bookings in train: ", df.loc[(df.booking_bool == 1)].shape[0])
        print("Clicks: ", df.loc[(df.click_bool == 1)].shape[0])
        print("Chance of booking: ", df.loc[(df.booking_bool == 1)].shape[0] / df[df.booking_bool.notnull()].shape[0] * 100, '%')
    

    return 


def normalize(df):
    pass


def plot_distr(df):
    plt.figure(figsize=(16, 8))
    sns.set(style="whitegrid")

    #Choose a random subset of IDs
    num_srch_ids_to_plot = 5000
    unique_srch_ids = df['prop_id'].unique()
    selected_srch_ids = np.random.choice(unique_srch_ids, num_srch_ids_to_plot, replace=False)

    #Filter to include only the selected search IDs
    subset_df = df[df['prop_id'].isin(selected_srch_ids)]

    #Create a box plot of prices grouped by search ID
    ax = sns.boxplot(x='prop_id', y='price_usd', data=subset_df)

    #Set the labels and title
    ax.set_xlabel('Property ID')
    ax.set_ylabel('Price (USD)')
    ax.set_title(f'Price Distribution by Property ID (Random Subset of {num_srch_ids_to_plot} Property IDs)')

    #Rotate the x-axis 
    plt.xticks(rotation=90)

    # Show the plot
    plt.show()

def preprocess(df):

    #Process the date/time feature by creating 2 new ftrs -> "month" and "year" and dropped the whole column eventually
    df['date_time'] = pd.to_datetime(df['date_time'])

    df['year'] = df['date_time'].dt.year
    df['month'] = df['date_time'].dt.month
    df = df.drop('date_time', axis=1)

    #More than 80-90% missing -> drop 
    df = df.drop(columns = ["visitor_hist_starrating", "visitor_hist_adr_usd", "srch_query_affinity_score"], axis=1)

    #Fill missing with -1 (we can say the difference between bad score and no data) ~1m missing values
    df['prop_location_score2'].fillna(-1, inplace=True)

    #TODO: Find what you will do with position (position bias)
    #Create target variable
    conditions = [
    (df['click_bool'] == 1) & (df['booking_bool'] == 1),
    (df['click_bool'] == 1) & (df['booking_bool'] == 0),
    (df['click_bool'] == 0) & (df['booking_bool'] == 0),
    ]

    choices = [2, 1, 0]

    df['suitability'] = np.select(conditions, choices, default=0)
    df = df.drop(columns = ["click_bool", "booking_bool", "gross_bookings_usd", "position"], axis=1)

    '''Creating "comp_rate_sum" which has the sum of the values comp{number}_rate of all competitors given prop_id (ignoring null values) ,
    "comp_inv_sum" which has the sum of the values comp{number}_inv of all competitors given prop_id (ignoring null values) and "comp_diff_aggr"
    which is the average absolute percentage difference with all competitors (counting null values as 0 )
    '''
    comp_rate_cols = [f'comp{i}_rate' for i in range(1, 9)]
    df['comp_rate_sum'] = df[comp_rate_cols].fillna(0).sum(axis=1)

    comp_inv_cols = [f'comp{i}_inv' for i in range(1, 9)]
    df['comp_inv_sum'] = df[comp_inv_cols].fillna(0).sum(axis=1)

    comp_diff_cols = [f'comp{i}_rate_percent_diff' for i in range(1, 9)]
    df['comp_diff_aggr'] = df[comp_diff_cols].fillna(0).abs().mean(axis=1)

    #Testing new columns
    print(df[['prop_id', 'comp_rate_sum', 'comp_inv_sum', 'comp_diff_aggr']].head())


def position_bias_plots(df):

    # Position x booking
    booked_count_by_position = df[df['booking_bool'] == 1].groupby('position')['booking_bool'].count()

    plt.figure(figsize=(12, 6))
    booked_count_by_position.plot(kind='bar')

    plt.xlabel('Position')
    plt.ylabel('Count of Booked Hotels')
    plt.title('Positions vs. Count of Booked Hotels')

    plt.show()

    # Position x clicked
    clicked_count_by_position = df[df['click_bool'] == 1].groupby('position')['click_bool'].count()

    plt.figure(figsize=(12, 6))
    clicked_count_by_position.plot(kind='bar')

    plt.xlabel('Position')
    plt.ylabel('Count of Clicked Hotels')
    plt.title('Positions vs. Count of Clicked Hotels')

    plt.show()
    


if __name__ == "__main__":
    #df_test = load_data("data/test_set_VU_DM.csv")
    df = load_data("data/training_set_VU_DM.csv")
    
    #print("Exploration of training set: \n")
    #explore_data(df)

    #print("#################### \n ################### \n")
    #print("Exploration of test set: \n")
    #explore_data(df_test,train = 0)

    print(pd.crosstab(df['click_bool'], df['booking_bool']))

    
    
    preprocess(df)

    #plot_distr(df)

    #df = process_datetime(df)

