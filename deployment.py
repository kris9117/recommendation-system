import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import pickle

data = pd.read_pickle('data.pkl')

model = pickle.load(open('kmeans.pkl', 'rb'))

song_vectorizer = CountVectorizer()
song_vectorizer.fit(data['genre'])


def get_similarities(song_name, df):
    text_array1 = song_vectorizer.transform(df[df['track_name'] == song_name]['genre']).toarray()

    num_array1 = df[df['track_name'] == song_name].select_dtypes(include=np.number).to_numpy()

    # We will store similarity for each row of the dataset.
    sim = []
    for idx, row in df.iterrows():
        name = row['track_name']
        # Getting vector for current song.
        text_array2 = song_vectorizer.transform(df[df['track_name'] == name]['genre']).toarray()

        num_array2 = df[df['track_name'] == name].select_dtypes(include=np.number).to_numpy()

        # Calculating similarities for text as well as numeric features
        text_sim = cosine_similarity(text_array1, text_array2)[0][0]
        num_sim = cosine_similarity(num_array1, num_array2)[0][0]
        sim.append(text_sim + num_sim)

    return sim


def recommend_songs(song_name, data=data):
    # Base case
    if data[data['track_name'] == song_name].shape[0] == 0:

        for song in data.sample(n=5)['track_name'].values:
            print(song)
        return
    data['similarity_factor'] = get_similarities(song_name, data)
    data.sort_values(by=['similarity_factor', 'popularity'],
                     ascending=[False, False],
                     inplace=True)
    # First song will be the input song itself as the similarity will be highest.
    return data[['track_name', 'genre']][2:7]


def main():
    st.title('Music Recommendation')

    st.selectbox(
        'search',
        data['track_name']
    )

    if st.button('Recommend'):
        st.write(recommend_songs(song_name=data['track_name']))


if __name__ == '__main__':
    main()
