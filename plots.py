import time
import argparse
import pickle
import os
import gc

import pandas
import numpy as np



def load_data(file_path):
    gc.collect()
    print("Loading data from file {} ...".format(file_path))
    df = pandas.read_csv(file_path)
    print("Finished loading data....")
    return df

def explore_data(df):
    print("Exectuting df.head() ...")
    print(df.head())

    df_desc = df.describe(include='object')

    # Use the Styler class to format the table
    '''styled_table = df_desc.style.set_table_attributes('style="font-size: 12px; border: 1px solid black;"') \
                .set_caption('Summary Statistics') \
                .set_properties(**{'text-align': 'center'}) \
                .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}]) 

    html = styled_table.render()
    with open('styled_table.html', 'w') as f:
        f.write(html)

    # Display the formatted table
    print(styled_table)'''
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

    return 

if __name__ == "__main__":
    '''parser = argparse.ArgumentParser("Trains the model and outputs the prediction.")
    parser.add_argument(
        "training_set_VU_DM" )
    parser.add_argumnt(
        "test_set_VU_DM",
    )
    parser.add_argument(
        "output",
    )
    args = parser.parse_args()
    train_csv = args.train_csv
    test_csv = args.test_csv
    output_dir = args.output_dir'''
    
    df = load_data("training_set_VU_DM.csv")
    explore_data(df)