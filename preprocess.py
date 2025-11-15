# preprocess_data.py

import pandas as pd
import numpy as np
import pickle

def preprocess_and_save():
    print("Loading raw datasets...")

    df1 = pd.read_csv('./combined_data_1.txt', header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1])
    df2 = pd.read_csv('C:/Users/jaihi/OneDrive/Desktop/Netflix_sys/archive/combined_data_2.txt', header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1])
    df3 = pd.read_csv('./archive/combined_data_3.txt', header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1])
    df4 = pd.read_csv('./archive/combined_data_4.txt', header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1])

    for df in [df1, df2, df3, df4]:
        df['Rating'] = df['Rating'].astype(float)

    df = pd.concat([df1, df2, df3, df4])
    df.index = np.arange(0, len(df))

    df_nan = pd.DataFrame(pd.isnull(df.Rating))
    df_nan = df_nan[df_nan['Rating'] == True].reset_index()

    movie_np = []
    movie_id = 1
    for i, j in zip(df_nan['index'][1:], df_nan['index'][:-1]):
        temp = np.full((1, i - j - 1), movie_id)
        movie_np = np.append(movie_np, temp)
        movie_id += 1
    last_record = np.full((1, len(df) - df_nan.iloc[-1, 0] - 1), movie_id)
    movie_np = np.append(movie_np, last_record)

    df = df[pd.notnull(df['Rating'])]
    df['Movie_Id'] = movie_np.astype(int)
    df['Cust_Id'] = df['Cust_Id'].astype(int)

    f = ['count', 'mean']
    df_movie_summary = df.groupby('Movie_Id')['Rating'].agg(f)
    movie_benchmark = round(df_movie_summary['count'].quantile(0.7), 0)
    drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

    df_cust_summary = df.groupby('Cust_Id')['Rating'].agg(f)
    cust_benchmark = round(df_cust_summary['count'].quantile(0.7), 0)
    drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index

    df = df[~df['Movie_Id'].isin(drop_movie_list)]
    df = df[~df['Cust_Id'].isin(drop_cust_list)]

    df_title_raw = pd.read_csv('./movie_titles.csv', encoding="ISO-8859-1", header=None, engine='python', on_bad_lines='skip')
    df_title_raw.columns = ['Movie_Id', 'Year', 'Name', *df_title_raw.columns[3:]]
    df_title_raw = df_title_raw[['Movie_Id', 'Year', 'Name']]

    df_p = pd.pivot_table(df, values='Rating', index='Cust_Id', columns='Movie_Id')

    # Save all processed components
    with open('preprocessed_data.pkl', 'wb') as f:
        pickle.dump((df, df_title_raw, df_movie_summary, df_p, drop_movie_list), f)

    print("✅ Data preprocessing completed and saved to preprocessed_data.pkl")

if __name__ == "__main__":
    preprocess_and_save()
