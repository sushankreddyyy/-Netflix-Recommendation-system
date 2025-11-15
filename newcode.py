import streamlit as st
import pandas as pd
import numpy as np
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
import pickle
import os

# Load data
@st.cache_data
def load_data():
    # Load all 4 parts of the Netflix dataset
    df1 = pd.read_csv('./combined_data_1.txt', header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1])
    df2 = pd.read_csv('C:/Users/jaihi/OneDrive/Desktop/Netflix_sys/archive/combined_data_2.txt', header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1])
    df3 = pd.read_csv('./archive/combined_data_3.txt', header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1])
    df4 = pd.read_csv('./archive/combined_data_4.txt', header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1])

    # Convert ratings to float
    for df in [df1, df2, df3, df4]:
        df['Rating'] = df['Rating'].astype(float)

    print('Dataset 1 shape: {}'.format(df1.shape))
    print('Dataset 2 shape: {}'.format(df2.shape))
    print('Dataset 3 shape: {}'.format(df3.shape))
    print('Dataset 4 shape: {}'.format(df4.shape))

    # Combine all datasets
    df = pd.concat([df1, df2, df3, df4])
    df.index = np.arange(0, len(df))

    # Identify NaN rows to assign Movie_Id
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

    # Filter NaNs and assign movie/user IDs
    df = df[pd.notnull(df['Rating'])]
    df['Movie_Id'] = movie_np.astype(int)
    df['Cust_Id'] = df['Cust_Id'].astype(int)

    # Filter popular movies and active users
    f = ['count', 'mean']
    df_movie_summary = df.groupby('Movie_Id')['Rating'].agg(f)
    movie_benchmark = round(df_movie_summary['count'].quantile(0.7), 0)
    drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

    df_cust_summary = df.groupby('Cust_Id')['Rating'].agg(f)
    cust_benchmark = round(df_cust_summary['count'].quantile(0.7), 0)
    drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index

    df = df[~df['Movie_Id'].isin(drop_movie_list)]
    df = df[~df['Cust_Id'].isin(drop_cust_list)]

    # Load movie titles
    df_title_raw = pd.read_csv('./movie_titles.csv', encoding="ISO-8859-1", header=None, engine='python', on_bad_lines='skip')
    df_title_raw.columns = ['Movie_Id', 'Year', 'Name', *df_title_raw.columns[3:]]
    df_title_raw = df_title_raw[['Movie_Id', 'Year', 'Name']]

    # Create pivot table for similarity
    df_p = pd.pivot_table(df, values='Rating', index='Cust_Id', columns='Movie_Id')

    return df, df_title_raw, df_movie_summary, df_p, drop_movie_list

df, df_title_raw, df_movie_summary, df_p, drop_movie_list = load_data()

st.title("🎬 Movie Recommender System with SVD & Correlation")

# Sidebar
st.sidebar.header("Options")
tab = st.sidebar.radio("Choose a functionality", ["User-based Recommendation", "Movie-based Similarity", "Train SVD Model"])

# Load or train SVD
@st.cache_resource
def load_or_train_svd_model():
    model_path = 'svd_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            svd = pickle.load(f)
        return svd
    else:
        reader = Reader()
        data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']], reader)
        trainset = data.build_full_trainset()
        svd = SVD()
        svd.fit(trainset)
        return svd

svd = load_or_train_svd_model()

# User-based Recommendation
if tab == "User-based Recommendation":
    st.header("🎯 Personalized Recommendations")
    user_id = st.number_input("Enter User ID", value=785314, step=1)

    user_df = df_title_raw.copy()
    user_df = user_df[~user_df['Movie_Id'].isin(drop_movie_list)]
    user_df['Estimate_Score'] = user_df['Movie_Id'].apply(lambda x: svd.predict(user_id, x).est)
    top_recommendations = user_df.sort_values('Estimate_Score', ascending=False).head(10)

    st.write("Top 10 Recommended Movies:")
    st.dataframe(top_recommendations[['Name', 'Estimate_Score']].reset_index(drop=True))

# Movie-based Similarity
elif tab == "Movie-based Similarity":
    st.header("🔗 Movie Similarity using Pearson Correlation")
    movie_title = st.selectbox("Select a movie", df_title_raw['Name'].dropna().sort_values().unique())
    min_count = st.slider("Minimum number of reviews for similar movies", 0, 100, 0)

    movie_row = df_title_raw[df_title_raw['Name'] == movie_title]
    if movie_row.empty:
        st.warning("Movie not found.")
    else:
        movie_id = movie_row['Movie_Id'].values[0]
        if movie_id not in df_p.columns:
            st.warning(f"Movie ID {movie_id} not found in rating matrix.")
        else:
            target = df_p[movie_id]
            similar_to_target = df_p.corrwith(target)
            corr_target = pd.DataFrame(similar_to_target, columns=['PearsonR']).dropna()
            corr_target = corr_target.sort_values('PearsonR', ascending=False)
            corr_target.index = corr_target.index.map(int)
            result = corr_target.join(df_title_raw.set_index('Movie_Id')).join(df_movie_summary)[['PearsonR', 'Name', 'count', 'mean']]
            filtered = result[result['count'] > min_count].head(10)

            st.write("Top 10 Similar Movies:")
            st.dataframe(filtered.reset_index(drop=True))

# Train SVD Model
elif tab == "Train SVD Model":
    st.header("📈 Train and Evaluate SVD")
    st.write("Evaluating the model using cross-validation...")

    reader = Reader()
    data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']], reader)
    results = cross_validate(SVD(), data, measures=['RMSE', 'MAE'], cv=3, verbose=False)

    st.write("### Evaluation Results")
    st.json({k: list(map(float, v)) for k, v in results.items()})

    if st.button("💾 Save Model"):
        with open('svd_model.pkl', 'wb') as f:
            pickle.dump(svd, f)
        st.success("Model saved to `svd_model.pkl`.")

