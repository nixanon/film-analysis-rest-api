import json
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra

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
print(find_film('brilliant gus gorman'))



#df = df.fillna('')
#df_json = find_film("Superman")

#print(df_json["title"])
#print(df_json['overview'])

#print(df.head(5))


# df1.columns = ['id','title','cast','crew']
# df2= df2.merge(df1,on='id')

# def preprocess(x):
#     #x = x.sub('[^a-z\s]', '', x.lower())                  # get rid of noise
#     x = [w for w in x.split() if w not in set(stopwords)]  # remove stopwords
#     return ' '.join(x)                                     # join the list

#Import TfIdfVectorizer from scikit-learn
#from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
#tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
# df2['overview'] = df2['overview'].fillna('')

#Lowercase the text in an overview
# df2['overview'] = df2['overview'].str.lower()

# from nltk.tokenize import RegexpTokenizer
# tokenizer = RegexpTokenizer(r'[A-Z]{2,3}[1-9][0-9]{3,3}')
# df2['tk_overview'] = df2['overview'].apply(tokenizer.tokenize)

# from nltk.corpus import stopwords
# stopwords = stopwords.words('english')
# df2['overview'] = df2['tk_overview'].apply(preprocess)
#content = [w for w in text if w.lower() not in stopwords]
# #Construct the required TF-IDF matrix by fitting and transforming the data
# tfidf_matrix = tfidf.fit_transform(df2['overview'])

# #Output the shape of tfidf_matrix
# tfidf_matrix.shape