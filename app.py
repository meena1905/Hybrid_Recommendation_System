import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import load_npz
import gdown
import os
import gdown

# Download predicted ratings
if not os.path.exists("predicted_ratings.npz"):
    gdown.download(
        "https://drive.google.com/uc?id=1M2AWIIC139vC7qlqvG448wLqL234l5PH",
        "predicted_ratings.npz",
        quiet=False
    )

# Download filtered data
if not os.path.exists("filtered_data.csv"):
    gdown.download(
        "https://drive.google.com/uc?id=1CVB1iqzcnZkJd2wE5Lc1kc86YmhkIMUv",
        "filtered_data.csv",
        quiet=False
    )

# --- Load data ---
movies = pd.read_csv("movies.csv")
filtered_data = pd.read_csv("filtered_data.csv")

# Content-based
tfidf_matrix = load_npz("tfidf_matrix.npz")

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

# Collaborative filtering
user_item_matrix = load_npz("user_item_matrix.npz")

data = np.load("predicted_ratings.npz")
predicted_ratings = data["predicted_ratings"]


# --- Content Based ---
def content_recommend(movie_title, top_n=10):

    matches = movies[movies['title'].str.lower().str.strip() == movie_title.lower().strip()]

    if matches.empty:
        return "Movie not found."

    idx = matches.index[0]

    sim_scores = tfidf_matrix[idx] @ tfidf_matrix.T
    sim_scores = np.array(sim_scores.todense()).ravel()

    top_indices = sim_scores.argsort()[::-1][1:top_n+1]

    return movies.iloc[top_indices][['title','genres']]


# --- Collaborative Filtering ---
def cf_recommend(user_index, top_n=10):

    scores = predicted_ratings[user_index].copy()

    watched = user_item_matrix[user_index].toarray().ravel() > 0
    scores[watched] = -np.inf

    top_indices = np.argsort(scores)[::-1][:top_n]

    return movies.iloc[top_indices][['title','genres']]


# --- Hybrid ---
def hybrid_recommend(user_index, movie_title, top_n=10, alpha=0.7):

    cf_scores = predicted_ratings[user_index].copy()

    watched = user_item_matrix[user_index].toarray().ravel() > 0
    cf_scores[watched] = -np.inf

    movie_idx = movies[movies['title']==movie_title].index[0]

    content_scores = tfidf_matrix[movie_idx] @ tfidf_matrix.T
    content_scores = np.array(content_scores.todense()).ravel()

    # FIX: make both same size
    content_scores = content_scores[:len(cf_scores)]

    hybrid_scores = alpha * cf_scores + (1-alpha) * content_scores

    hybrid_scores[watched] = -np.inf

    top_indices = np.argsort(hybrid_scores)[::-1][:top_n]

    return movies.iloc[top_indices][['title','genres']]

# --- Streamlit UI ---

st.title("🎬 Hybrid Movie Recommendation System")

option = st.radio(
    "Choose Recommendation Type",
    ("Content-based","Collaborative","Hybrid")
)


# --- Content ---
if option == "Content-based":

    movie_input = st.selectbox(
        "Select a Movie",
        movies['title'].values
    )

    if st.button("Recommend"):
        st.write(content_recommend(movie_input))


# --- Collaborative ---
elif option == "Collaborative":

    user_input = st.number_input(
        "Enter user index",
        min_value=0,
        max_value=user_item_matrix.shape[0]-1
    )

    if st.button("Recommend"):
        st.write(cf_recommend(int(user_input)))


# --- Hybrid ---
elif option == "Hybrid":

    user_input = st.number_input(
        "Enter user index",
        min_value=0,
        max_value=user_item_matrix.shape[0]-1
    )

    movie_input = st.selectbox(
        "Select a Movie",
        movies['title'].values
    )

    if st.button("Recommend"):
        st.write(hybrid_recommend(int(user_input), movie_input))