import json
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra

import nltk
import math
# from nltk.tokenize import RegexpTokenizer
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer 
#df1 = pd.read_csv('./tmdb_5000_credits.csv')

def find_film(name):
    try:
        name = name.lower()
        df['title']= df['title'].astype(str).str.lower()
        df['overview'] = df['overview'].astype(str).str.lower()
        df['text'] = df['title'] + df['overview']
        films = df.loc[df['text'].str.contains(name)] 

        films = films[['budget', 'title', 'id', 'overview', 'release_date']]
        #print(films.head())
        row_count = df.shape[0] # number of rows in dataframe
        if row_count < 5:
            films = films.head(row_count) 
        else:
            films = films.head()# save first 5 rows
        
        #films_dict = films.set_index('title').T.to_dict()
        films_dict = films.to_dict('records')
        return films_dict
        #print(films_dict)
        
        
    except : 
        return None

df = pd.read_csv('./tmdb_5000_movies.csv')
print(find_film('superman'))
