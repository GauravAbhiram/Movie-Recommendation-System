#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[3]:


df = pd.read_csv(r"C:\Users\gaura\Downloads\movies_metadata.csv")

# Keep only required columns
df = df[['title', 'genres', 'overview']]
df.dropna(inplace=True)

df.head()


# In[3]:


df.drop(columns=['poster_path'], inplace=True)


# In[4]:


df.describe()


# In[5]:


print(df)


# In[8]:


print(df.head)


# In[9]:


df.iloc[0:4]


# In[7]:


df['combined_features'] = df['genres'] + " " + df['overview']


# In[12]:


df.iloc[0:4]


# In[8]:


tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(df['combined_features'])


# In[16]:


cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# In[9]:


tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

tfidf_matrix = tfidf.fit_transform(df['combined_features'])


# In[16]:


tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

tfidf_matrix = tfidf.fit_transform(df['combined_features'])

from sklearn.metrics.pairwise import cosine_similarity

def recommend_movies(movie_title, num_recommendations=5):
    if movie_title not in df['title'].values:
        return "Movie not found in dataset"

    idx = df[df['title'] == movie_title].index[0]
    df.reset_index(drop=True, inplace=True)

    # compute similarity only for one movie
    similarity_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()
    
    # get top similar movies
    similar_indices = similarity_scores.argsort()[::-1][1:num_recommendations+1]
    
    return df['title'].iloc[similar_indices]


# In[19]:


print(recommend_movies("RRR", 5))


# In[18]:


from collections import Counter
import ast

def extract_genres_list(genres_str):
    genres = ast.literal_eval(genres_str)
    return [g['name'] for g in genres]

df['genres_list'] = df['genres'].apply(extract_genres_list)


# In[19]:


genre_counts = Counter(
    genre for genres in df['genres_list'] for genre in genres
)


# In[23]:


genres, counts = zip(*genre_counts.most_common(10))

import matplotlib.pyplot as plt

plt.figure()
plt.bar(genres, counts)
plt.xticks(rotation=45)
plt.title("Top 10 Movie Genres")
plt.xlabel("Genre")
plt.ylabel("Number of Movies")
plt.show()


# In[22]:


print(genre_counts)


# In[27]:


import numpy as np

feature_names = tfidf.get_feature_names()
tfidf_scores = tfidf_matrix.sum(axis=0).A1

top_indices = np.argsort(tfidf_scores)[::-1][:10]
top_words = [feature_names[i] for i in top_indices]
top_scores = tfidf_scores[top_indices]
import matplotlib.pyplot as plt
plt.figure()
plt.barh(top_words[::-1], top_scores[::-1])
plt.title("Top TF-IDF Keywords")
plt.xlabel("TF-IDF Score")
plt.show()


# In[12]:


import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Compute similarity scores for one reference movie (e.g., first movie)
similarity_scores = cosine_similarity(
    tfidf_matrix[0],
    tfidf_matrix
).flatten()

# Plot histogram
plt.figure()
plt.hist(similarity_scores, bins=30)
plt.title("Distribution of Cosine Similarity Scores")
plt.xlabel("Cosine Similarity Score")
plt.ylabel("Number of Movies")
plt.show()


# In[13]:


import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Select a movie
movie_name = "Inception"

# Get index of the movie
idx = df[df['title'] == movie_name].index[0]

# Compute cosine similarity
similarity_scores = cosine_similarity(
    tfidf_matrix[idx],
    tfidf_matrix
).flatten()

# Get top 5 similar movies (excluding itself)
top_indices = similarity_scores.argsort()[::-1][1:6]
top_movies = df['title'].iloc[top_indices]
top_scores = similarity_scores[top_indices]

# Plot bar chart
plt.figure()
plt.bar(top_movies, top_scores)
plt.xticks(rotation=45)
plt.title(f"Top Similar Movies to {movie_name}")
plt.xlabel("Movie")
plt.ylabel("Cosine Similarity Score")
plt.show()


# In[14]:


df['overview_length'] = df['overview'].apply(lambda x: len(x.split()))

plt.figure()
plt.hist(df['overview_length'], bins=30)
plt.title("Distribution of Movie Description Length")
plt.xlabel("Number of Words")
plt.ylabel("Number of Movies")
plt.show()


# In[ ]:




