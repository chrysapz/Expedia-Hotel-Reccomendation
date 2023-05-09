import time
import argparse
import pickle
import os
import gc
import lightgbm
from lightgbm import early_stopping, log_evaluation

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
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

    #Remove when training as whole
    #df = df.sample(n=1000000, random_state=42)

    print("--Data Loaded--")
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

def to_boolean(x):
    return True if x==1 else False if x==0 else None

def preprocess(df, train=True):
    gc.collect()

    #fill with -1 to differentiate between missing and actual ~1.6 million missing vs
    df['orig_destination_distance'] = df['orig_destination_distance'].fillna(-1)

    #Remove rows where attribute "prop_review_score" is missing
    df = df.dropna(subset=['prop_review_score'])

    #Turn columns that should be boolean
    for col in ["prop_brand_bool", "promotion_flag", "srch_saturday_night_bool"]:
        df.loc[:, col] = df[col].apply(to_boolean)

    # Convert the date_time column (doing it this way to avoid a stupid error)
    df_processed = df.copy()
    df_processed['date_time'] = pd.to_datetime(df_processed['date_time'])

    #df_processed['year'] = df_processed['date_time'].dt.year
    df_processed['month'] = df_processed['date_time'].dt.month

    df_processed = df_processed.drop('date_time', axis=1)    
    df = df_processed

    #difference between price_usd and visitor_hist_adr_usd => identify if a hotel is more or less expensive compared to what the visitor usually books
    #a positive value indicates the hotel has a higher price than the customer's past bookings, while a negative value indicates the opposite
    df['price_diff_visitor_history'] = df['price_usd'] - df['visitor_hist_adr_usd']

    #difference prop_starrating and visitor_hist_starrating=>identify if a hotel has a higher or lower star rating compared to what the visitor usually books
    #a positive value indicates the hotel has a higher star rating than the customer's past choices, while a negative value indicates a lower rating
    df['star_diff_visitor_history'] = df['prop_starrating'] - df['visitor_hist_starrating']

    #More than 80-90% missing -> drop 
    df = df.drop(columns = ["visitor_hist_starrating", "visitor_hist_adr_usd", "srch_query_affinity_score"], axis=1)

    #Initially I did that: Fill missing with -1 (we can say the difference between bad score and no data) ~1m missing values
    #Then I did the aggregated feature and this no longer made sense so -> 0
    df['prop_location_score2'].fillna(0, inplace=True)


    #AGGREGATED FEATURES

    '''Creating "comp_rate_sum" which has the sum of the values comp{number}_rate of all competitors given prop_id (ignoring null values) ,
    "comp_inv_sum" which has the sum of the values comp{number}_inv of all competitors given prop_id (ignoring null values) and "comp_diff_aggr"
    which is the average absolute percentage difference with all competitors (counting null values as 0 )
    '''
    comp_rate_cols = [f'comp{i}_rate' for i in range(1, 9)]
    df['comp_rate_sum'] = df[comp_rate_cols].fillna(0).sum(axis=1)
    df = df.drop(columns = [f'comp{i}_rate' for i in range(1, 9)], axis=1)

    comp_inv_cols = [f'comp{i}_inv' for i in range(1, 9)]
    df['comp_inv_sum'] = df[comp_inv_cols].fillna(0).sum(axis=1)
    df = df.drop(columns = [f'comp{i}_inv' for i in range(1, 9)], axis=1)

    comp_diff_cols = [f'comp{i}_rate_percent_diff' for i in range(1, 9)]
    df['comp_diff_aggr'] = df[comp_diff_cols].fillna(0).abs().mean(axis=1)
    df = df.drop(columns = [f'comp{i}_rate_percent_diff' for i in range(1, 9)], axis=1)

    #Testing new columns
    #print(df[['prop_id', 'comp_rate_sum', 'comp_inv_sum', 'comp_diff_aggr']].head())

    

    #ratio between price_usd and prop_log_historical_price =>identify if a hotel is currently more or less expensive compared to its historical average
    #since the historical price is given in logarithmic form => exponentiate it to convert it back to its original scale before computing ratio
    #value > 1 indicates the current price is higher than the historical average else the current price is lower.
    df['price_ratio_prop_history'] = df['price_usd'] / np.exp(df['prop_log_historical_price'])

    #aggregate into single score, not sure if this avg thing im doing tho makes sense
    #df['combined_location_score'] = (df['prop_location_score1'] + df['prop_location_score2']) / 2

    #self-explanatory
    df['price_per_adult'] = df['price_usd'] / df['srch_adults_count']

    ##TEMPORAL FEATURESS##
    '''Goal: identify temporal trends'''

    avg_price_month = df.groupby('month')['price_usd'].agg('mean') #seasonal trends in pricing
    #avg_price_year = df.groupby('year')['price_usd'].agg('mean') #yearly trends i.e. rising prices
    avg_review_score_month = df.groupby('month')['prop_review_score'].agg('mean') #seasonality in customer satisfaction but might be useless (could be done for years)
    if train:
        monthly_booking_rate = df.groupby('month')['booking_bool'].agg('mean') #seasonal patterns in booking behavior (could also be done for years)
        df = df.merge(monthly_booking_rate, on='month', suffixes=('', '_booking_rate'))

    df = df.merge(avg_price_month, on='month', suffixes=('', '_avg_month'))
    df = df.merge(avg_review_score_month, on='month', suffixes=('', '_avg_review_month'))
    #df = df.merge(avg_price_year, on='year', suffixes=('', '_avg_year'))
    

    ##GROUP-AGGREGATED FEATURES##
    
    #avg review score for each property and destination combination => identify if some properties have better reviews in certain locations.
    avg_review_score_per_property_dest = df.groupby(['prop_id', 'srch_destination_id'])['prop_review_score'].mean().reset_index()
    avg_review_score_per_property_dest.columns = ['prop_id', 'srch_destination_id', 'avg_review_score_per_property_dest']
    df = df.merge(avg_review_score_per_property_dest, on=['prop_id', 'srch_destination_id'], how='left')

    #booking rate per property => identify popular properties but not sure if it fucks up training because it kinda incorporates book_bool
    #if we decide its useful we can do the same with clicks or aggregate clicks and bookings and do both
    if train:
        booking_rate_per_property = df.groupby('prop_id')['booking_bool'].mean().reset_index()
        booking_rate_per_property.columns = ['prop_id', 'booking_rate_per_property']
        df = df.merge(booking_rate_per_property, on='prop_id', how='left')


    location_avg_price = df.groupby('prop_country_id')['price_usd'].agg('mean')
    df = df.merge(location_avg_price, on='prop_country_id', suffixes=('', '_avg_location'))

    #Feature normalization
    features_to_normalize = [
    'price_diff_visitor_history',
    'price_ratio_prop_history',
    "avg_review_score_per_property_dest",
    "price_usd_avg_location",
    "prop_starrating", 
    "prop_review_score",
    "prop_location_score1",
    "prop_location_score2",
    "prop_log_historical_price",
    "price_usd",
    "promotion_flag",
    "orig_destination_distance"]

    scaler = RobustScaler()
    df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])


    if train:
        conditions = [
        (df['click_bool'] == 1) & (df['booking_bool'] == 1),
        (df['click_bool'] == 1) & (df['booking_bool'] == 0),
        (df['click_bool'] == 0) & (df['booking_bool'] == 0),
        ]

        choices = [2, 1, 0]

        df['suitability'] = np.select(conditions, choices, default=0)
        df = df.drop(columns = ["click_bool", "booking_bool", "gross_bookings_usd", "position"], axis=1)
        y = df["suitability"]
        X = df.drop(columns = ["suitability"], axis =1 )

        return X,y
    else:
        X=df
        return X, None
    

def position_bias_plots(df):

    #Position x booking
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
    

def split(X, y, val_start, val_stop):
        
    x_new = pd.concat([X[0:val_start], X[val_stop:]])
    y_new = np.concatenate((y[0:val_start], y[val_stop:]), axis=0)
    x_out = X[val_start:val_stop]
    y_out = y[val_start:val_stop]

    #group query data for the Ranker 
    groups = x_new["srch_id"].value_counts(sort=False).sort_index()
    val_groups = x_out["srch_id"].value_counts(sort=False).sort_index()
    #print("Inside split: \n")
    #print(groups)
    #print(y_new)

    x_new = x_new.drop(columns = ["srch_id", "prop_id"], axis=1)
    x_out = x_out.drop(columns = ["srch_id", "prop_id"], axis=1)


    return (x_new, y_new, x_out, y_out, groups, val_groups) 

def train(
    x_new, y_new, x_out, y_out, boost_method, lr, groups, val_groups,
):

    ranker = lightgbm.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=1000,#2000,
        learning_rate=lr,
        label_gain=[0, 1, 2],
        random_state=42,
        boosting=boost_method,
    )

    gc.collect()
    early_stopping_callback = early_stopping(stopping_rounds=150, first_metric_only=True)
    log_evaluation_callback = log_evaluation(period=20)
    #print("The features used for training are: {}".format(x_new.columns.values))
    ranker.fit(
        x_new,
        y_new,
        eval_set=[(x_new, y_new), (x_out, y_out)],
        eval_group=[groups, val_groups],
        group=groups,
        eval_at=5,
        callbacks=[early_stopping_callback, log_evaluation_callback]
    )

    return ranker

def test(X_test, output_dir):

    gc.collect()

    model = pickle.load(open(os.path.join(output_dir, "model.dat"), "rb"))

    print("Test set has columns: {}".format(X_test.columns.values))
    prediction = model.predict(X_test)

    pred_csv_format = X_test[["srch_id", "prop_id"]]
    pred_csv_format["predicted"] = prediction
    pred_csv_format = pred_csv_format.sort_values(["srch_id", "predicted"], ascending=True)
    pred_csv_format[["srch_id", "prop_id"]].to_csv(os.path.join(output_dir, "prediction.csv"), index=False)

    print("Saved .csv")



def run(X, y, output_dir):
    
    
    boost_method = "gbdt" #'dart', 'rf' , 'goss'
    val_n = 75000
    lr = 0.1
    best_score=0
    name_of_best = ""
    for i in range(0, int(X.shape[0] / val_n)): # num of entries / val number
    #for i in range(0,1):
        name = str(int(time.time())) 
        print("Iteration of cross-validation: ", i, "\n")
        val_start = i * val_n
        val_end = (i + 1) * val_n
        x_new, y_new, x_out, y_out, groups, val_groups = split( 
            X, y, val_start, val_end
        )
        model = train(
            x_new, y_new, x_out, y_out, boost_method,lr, groups, val_groups, 
        )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        gc.collect()

        if i == 0 or (model.best_score_["valid_1"]["ndcg@5"] > best_score):
            name_of_best = name
            best_score = model.best_score_["valid_1"]['ndcg@5']
            pickle.dump(model, open(os.path.join(output_dir, "model.dat"), "wb"))
        

    test_data = load_data("data/test_set_VU_DM.csv")
    X_test, _ = preprocess(test_data, train=False)
    test(X_test, output_dir)



if __name__ == "__main__":
    #df_test = load_data("data/test_set_VU_DM.csv")
    df = load_data("data/training_set_VU_DM.csv")
    
    #print("Exploration of training set: \n")
    #explore_data(df)

    #print("#################### \n ################### \n")
    #print("Exploration of test set: \n")
    #explore_data(df_test,train = 0)

    #print(pd.crosstab(df['click_bool'], df['booking_bool']))

    '''df['prop_location_score1'].hist(bins=50)
    plt.xlabel('prop_location_score1')
    plt.ylabel('Frequency')
    plt.title('prop_location_score1')
    plt.show() 

    df['prop_location_score2'].hist(bins=50)
    plt.xlabel('prop_location_score2')
    plt.ylabel('Frequency')
    plt.title('prop_location_score2')
    plt.show() '''

    output_dir = "output/"
    

    train_X,train_y = preprocess(df)
    run(train_X,train_y, output_dir)

    #test_X, test_y = preprocess(df, train=False)
    #plot_distr(df)

    #df = process_datetime(df)

