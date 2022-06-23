# %% [markdown]
# # **Building Music Recommendation System using Spotify Dataset**
# 
# ![image.png](attachment:image.png)
# 
# Hello and welcome to my kernel. In this kernel, I have created Music Recommendation System using Spotify Dataset. To do this, I presented some of the visualization processes to understand data and done some EDA(Exploratory Data Analysis) so we can select features that are relevant to create a Recommendation System.

# %% [markdown]
# # **Import Libraries**

# %%
import os
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import plotly.express as px 
import matplotlib.pyplot as plt
import skimage
import streamlit.components.v1 as components
import base64
import datetime
from urllib.parse import urlencode
import requests
import re
import spotipy
import sys
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
from skimage import io
import warnings
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Music Recommendation Sytem", layout="wide")

# %% [markdown]
# # **Read Data**

# %%
data = pd.read_csv("data.csv")
genre_data = pd.read_csv('data_by_genres.csv')
year_data = pd.read_csv('data_by_year.csv')

# %%
print(data.info())

# %%
print(genre_data.info())

# %%
print(year_data.info())

# %% [markdown]
# We are going to check for all the analysis with the target as **'popularity'**. Before going to do that let's check for the Feature Correlation by considering a few features and for that, I'm going to use the **yellowbrick** package. You can learn more about it from the [documentation](https://www.scikit-yb.org/en/latest/index.html).

# %%
from yellowbrick.target import FeatureCorrelation

feature_names = ['acousticness', 'danceability', 'energy', 'instrumentalness',
       'liveness', 'loudness', 'speechiness', 'tempo', 'valence','duration_ms','explicit','key','mode','year']

X, y = data[feature_names], data['popularity']

# Create a list of the feature names
features = np.array(feature_names)

# Instantiate the visualizer
visualizer = FeatureCorrelation(labels=features)

plt.rcParams['figure.figsize']=(20,20)
visualizer.fit(X, y)     # Fit the data to the visualizer
visualizer.show()
st.write(visualizer)

# %% [markdown]
# # **Data Understanding by Visualization and EDA**

# %% [markdown]
# # **Music Over Time**
# 
# Using the data grouped by year, we can understand how the overall sound of music has changed from 1921 to 2020.

# %%
def get_decade(year):
    period_start = int(year/10) * 10
    decade = '{}s'.format(period_start)
    return decade

data['decade'] = data['year'].apply(get_decade)

fig = plt.figure(figsize=(10, 4))
sns.set(rc={'figure.figsize':(10 ,4)})
sns.countplot(data['decade'])
st.pyplot(fig)

# %%
sound_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']
fig = px.line(year_data, x='year', y=sound_features)

st.write(fig)

# %% [markdown]
# # **Characteristics of Different Genres**
# 
# This dataset contains the audio features for different songs along with the audio features for different genres. We can use this information to compare different genres and understand their unique differences in sound.

# %%
top10_genres = genre_data.nlargest(10, 'popularity')

fig = px.bar(top10_genres, x='genres', y=['valence', 'energy', 'danceability', 'acousticness'], barmode='group')

st.write(fig)

# %% [markdown]
# # **Clustering Genres with K-Means**
# 
# Here, the simple K-means clustering algorithm is used to divide the genres in this dataset into ten clusters based on the numerical audio features of each genres.

# %%
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=10))])
X = genre_data.select_dtypes(np.number)
cluster_pipeline.fit(X)
genre_data['cluster'] = cluster_pipeline.predict(X)

# %%
# Visualizing the Clusters with t-SNE

from sklearn.manifold import TSNE

tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
genre_embedding = tsne_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
projection['genres'] = genre_data['genres']
projection['cluster'] = genre_data['cluster']

fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genres'])

st.write(fig)

# %% [markdown]
# # **Clustering Songs with K-Means**

# %%
song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                  ('kmeans', KMeans(n_clusters=20, 
                                   verbose=False, n_init=4))
                                 ], verbose=False)

X = data.select_dtypes(np.number)
number_cols = list(X.columns)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
data['cluster_label'] = song_cluster_labels

# %%
# Visualizing the Clusters with PCA

from sklearn.decomposition import PCA

pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = data['name']
projection['cluster'] = data['cluster_label']

fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'])

st.write(fig)

# %% [markdown]
# # **Build Recommender System**
# 
# * Based on the analysis and visualizations, it’s clear that similar genres tend to have data points that are located close to each other while similar types of songs are also clustered together.
# * This observation makes perfect sense. Similar genres will sound similar and will come from similar time periods while the same can be said for songs within those genres. We can use this idea to build a recommendation system by taking the data points of the songs a user has listened to and recommending songs corresponding to nearby data points.
# * [Spotipy](https://spotipy.readthedocs.io/en/2.16.1/) is a Python client for the Spotify Web API that makes it easy for developers to fetch data and query Spotify’s catalog for songs. You have to install using `pip install spotipy`
# * After installing Spotipy, you will need to create an app on the [Spotify Developer’s page](https://developer.spotify.com/) and save your Client ID and secret key.

# %% [markdown]
# ### **Spotify authorizations**

# %%
#client id and secret for my application
client_id = '359aa96093614d0a9af9c0ddb30a1671'
client_secret= 'b39963440eae4f748c7e3accb89ebe35'
scope = 'user-library-read'

# if len(sys.argv) > 1:
#     username = sys.argv[1]
# else:
#     print("Usage: %s username" % (sys.argv[0],))
#     sys.exit()


auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)
#token = util.prompt_for_user_token(scope, client_id= client_id, client_secret=client_secret, redirect_uri='http://localhost:8881/')
#sp = spotipy.Spotify(auth=token)

class SpotifyAPI(object):
  access_token = None
  access_token_expires = datetime.datetime.now()
  access_token_did_expire = True
  client_id = None
  client_secret = None
  token_url = "https://accounts.spotify.com/api/token"

  def __init__(self, client_id, client_secret, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.client_id = client_id
    self.client_secret = client_secret

#We will then need to use our Client ID and Client Secret to create a base64 encoded string for our client credentials. 
# Encoding our credentials into a base64 string makes it more secure.
def get_client_credentials(self):
      
      #Returns a base64 encoded string
      
    client_id = self.client_id
    client_secret = self.client_secret
    if client_secret == None or client_id == None:
        raise Exception("You must set client_id and client_secret")
    client_creds = f"{client_id}:{client_secret}"
    client_creds_b64 = base64.b64encode(client_creds.encode())
    return client_creds_b64.decode()

def get_token_headers(self):
    client_creds_b64 = self.get_client_credentials()
    return {
      "Authorization": f"Basic {client_creds_b64}" 
    }

def get_token_data(self):
    return {
      "grant_type": "client_credentials"
    }

def perform_auth(self):
    token_url = self.token_url
    token_data = self.get_token_data()
    token_headers = self.get_token_headers()
    r = requests.post(token_url, data=token_data, headers=token_headers)
    if r.status_code not in range(200, 299): 
        raise Exception("Could not authenticate client")
        #return False
    data = r.json()
    now = datetime.datetime.now()
    access_token = data['access_token']
    expires_in = data['expires_in'] # seconds
    expires = now + datetime.timedelta(seconds=expires_in)
    self.access_token = access_token
    self.access_token_expires = expires 
    self.access_token_did_expire = expires < now
    return True

def get_access_token(self):
    token = self.access_token
    expires = self.access_token_expires
    now = datetime.datetime.now()
    if expires < now:
        self.perform_auth()
        return self.get_access_token()
    elif token == None:
        self.perform_auth()
        return self.get_access_token()
    return token

def get_resource_header(self):
    access_token = self.get_access_token()
    headers = {
        "Authorization": f"Bearer {access_token}"
    }      
    return headers

# %% [markdown]
# ##### **Credentials and tokens**

# %%
spotify = SpotifyAPI(client_id, client_secret)
access_token = spotify.access_token

headers = {
    "Authorization": f"Bearer {access_token}"
}

# %%
def get_year_track(name, artist):
    track = sp.search(q='artist: {} track: {}'.format(artist,name), type='track',limit=1)
    track_date=track['tracks']['items'][0]['album']['release_date']
    track_date=datetime.datetime.strptime(track_date, "%Y-%m-%d").year
    return(track_date)


# %%

#entrée: nom string, year string
#sortie: dictionnaire des spécificités du son correspondant au titre name et à l'année year
def find_song(name, artist):
    song_data = defaultdict()
    track = sp.search(q='artist: {} track: {}'.format(artist,name), type='track',limit=1)
    if track['tracks']['items'] == []:
        return None

    track = track['tracks']['items'][0]
    track_id = track['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = get_year_track(name,artist)

    song_data['explicit'] = [int(track['explicit'])]
    song_data['duration_ms'] = [track['duration_ms']]
    song_data['popularity'] = [track['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)

# %%
from collections import defaultdict
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import difflib

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

#on tente d'avoir les données du son via soit notre source de données soit via l'API spotify
def get_song_data(song, spotify_data):
    
    try:
        song_data = spotify_data.loc[(spotify_data['name'] == song['name']) 
                                & (spotify_data['year'] == song['year']) 
                                & (spotify_data['artists'].str.contains(song['artist'],case=False))].iloc[0]
        return song_data
    
    except IndexError:
        return find_song(song['name'], song['artist'])
        

def get_mean_vector(song_list, spotify_data):
    
    song_vectors = []
    
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)  
    
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):
    
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
            
    return flattened_dict


def recommend_songs( song_list, spotify_data, n_songs=10):
    
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)
    
    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')

# %%
# recommend_songs([{'name': "Dont Speak",'artist':"No doubt", 'year':1996}],  data)
#song_data=get_song_data([{'name': "Dont Speak",'artist':"No doubt", 'year':1996}], data)
# song_data = data[(data['name'] == 'Dont Speak') & (data['year'] == 1996) & ('No doubt' in data['artists'])].iloc[0]
# search_values=['Sergei Rachmaninoff', 'James Levine', 'Berliner Philharmoniker']
# d=dict(zip(['artists'],search_values))
# print(d)
# song_data = data[(data['artists'].isin(d))]
# print(type(data['artists'][0]))
# print((data['artists'][0]))
# song_data = data.loc[data['artists'].str.contains('Sergei Rachmaninoff',case=False)]
# # song_data = data.loc[(data['name'] == 'Piano Concerto No. 3 in D Minor, Op. 30: III. Finale. Alla breve') & (data['year'] == 1996) & (data['artists'].str.contains("Dont Speak",case=False))]
# song_data = data.loc[(data['year'] == 1996) & (data['artists'].str.contains("o",case=False))]
# song_data.iloc[0]
# print(song_data)
#song_center = get_mean_vector([{'name': "Dont Speak",'artist':"No doubt", 'year':1996}], data)
# scaler = song_cluster_pipeline.steps[0][1]
# scaled_song_center = scaler.transform(song_center.reshape(1, -1))

# data['artists'][0]


# %% [markdown]
# * This last cell will gives you a recommendation list of songs like this,
# 
# 
# ```
# [{'name': 'Life is a Highway - From "Cars"',
#   'year': 2009,
#   'artists': "['Rascal Flatts']"},
#  {'name': 'Of Wolf And Man', 'year': 1991, 'artists': "['Metallica']"},
#  {'name': 'Somebody Like You', 'year': 2002, 'artists': "['Keith Urban']"},
#  {'name': 'Kayleigh', 'year': 1992, 'artists': "['Marillion']"},
#  {'name': 'Little Secrets', 'year': 2009, 'artists': "['Passion Pit']"},
#  {'name': 'No Excuses', 'year': 1994, 'artists': "['Alice In Chains']"},
#  {'name': 'Corazón Mágico', 'year': 1995, 'artists': "['Los Fugitivos']"},
#  {'name': 'If Today Was Your Last Day',
#   'year': 2008,
#   'artists': "['Nickelback']"},
#  {'name': "Let's Get Rocked", 'year': 1992, 'artists': "['Def Leppard']"},
#  {'name': "Breakfast At Tiffany's",
#   'year': 1995,
#   'artists': "['Deep Blue Something']"}]
# ```
# 
# 
# 
# * You can change the given songs list as per your choice.

# %%
def visualize_songs(recommended_songs):
    with st.container():
        st.write("Here are your recommended songs ! ")
        col1, col2, col3 = st.columns([2,1,2])

        for i in range(len(recommended_songs)):
            artist=recommended_songs[i]['artists'][0]
            track= recommended_songs[i]['name']
            year_song=get_year_track(track,artist)
            dict_track=sp.search(q='artist: {} track: {} year: {}'.format(artist,track,year_song),type='track')
            track_id=dict_track['tracks']['items'][0]['id']
            open_url_track = """<iframe src=https://open.spotify.com/embed/track/{} width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""".format(track_id)
            if i%2==0:
                with col1:
                    components.html(
                        open_url_track,
                        height=400,
                    )
            else:
                with col3:
                    components.html(
                        open_url_track,
                        height=400,
                    )

with st.container():
    st.title(" Here's our recommendation system")
    st.write(" You can enter here a song title and its artist ")
    title_song = st.text_input("Song title")
    artist_song = st.text_input("Song artist")
    year_song = st.selectbox('Year',range(1900,2022)) 
    button_clicked = st.button("Give me related songs !")
    if button_clicked:
        song_data = get_song_data({'name':title_song,'artist':artist_song,'year':year_song}, data)
        if song_data is not None:
            dict_track=sp.search(q='artist: {} track: {} year: {}'.format(artist_song,title_song,year_song),type='track')
            track_id=dict_track['tracks']['items'][0]['id']
            open_url_track = """<iframe src=https://open.spotify.com/embed/track/{} width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""".format(track_id)
            components.html(
            open_url_track,
            height=400,
            )
            visualize_songs(recommend_songs([{'name': title_song, 'artist': artist_song, 'year':year_song}],  data))
        else:
            print('Warning: {} does not exist in Spotify nor in our database, please try another one'.format(title_song))

        

# visualize_songs(recommend_songs([{'name': "Dont Speak", 'year':1996}],  data))
# dict_track=sp.search(q='artist:' + 'No Doubt' + ' track:' + 'Dont Speak', type='track')
# print(dict_track['tracks']['items'][0]['album']['images'][0]['url'])
# print(dict_track['tracks']['items'][0].keys())
# print(dict_track['tracks']['items'][0]['external_urls']['spotify'])
# print(dict_track['tracks']['items'][0]['id'])



# %%
dict_track=sp.search(q='artist: {} track: {} year: {}'.format('NTO','Trauma',2018),type='track')
track_date=dict_track['tracks']['items'][0]['album']['release_date']
print(track_date)
track_date=datetime.datetime.strptime(track_date, "%Y-%m-%d").year
print(track_date)



