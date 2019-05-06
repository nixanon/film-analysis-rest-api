import json, nltk, math # data format, text processing, calculations
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import string
import time
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from math import log10, sqrt
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle
import re

df_movies = pd.read_csv('./tmdb_5000_movies.csv')
df_movies = df_movies.drop(columns=['runtime','spoken_languages', 'tagline','production_companies', 
'production_countries', 'original_title', 'original_language', 'keywords', 'status'])

categories = [ 'Western', 'Mystery', 'Foreign', 'History', 'Thriller', 'TV Movie', 'Drama', 
            'Horror', 'Adventure', 'Documentary', 'Crime', 'Comedy', 'Family', 'War', 'Romance', 
            'Fantasy', 'Animation', 'Action', 'Music', 'Science Fiction', 'tokens' ] # all of the genres used by the dataset

df_movies = df_movies.reindex( df_movies.columns.union(categories), axis=1 )
del categories[-1]
for c in categories:
    df_movies[c] = 0

class Film:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
        self.tf_idfs = {}                        # tf-idf vectors
        self.term_freqs = {}                     # (tfs) term frequencies of all tokens in all documents
        self.genre_term_freqs = {}
        self.models = []
        for c in categories:
            self.genre_term_freqs[c] = Counter()
        self.genre_freqs = Counter()             # keep track of all the occurences of each genre
        self.doc_freqs = Counter()               # document frequencies
        self.doc_lens = Counter()                # document lengths
        self.postings_list = {}                  # a sorted list in which each element is in the form (doc d, tf-idf weight w)
        self.stopword_list = set( stopwords.words('english') ).union( list(string.punctuation) )    
        self.df_genres = pd.DataFrame(columns = ['title'])
        start = time.time()
        try:
            global df_movies
            df_movies = df_movies.apply(self.process, axis=1) # perform additonal processing
            print(self.genre_freqs)
            self.calcTFIDF()
            self.normWeights()
            # Define a pipeline combining a text feature extractor with multi label classifier
            self.train, self.test = train_test_split(df_movies, random_state=42, test_size=0.25, shuffle=True)
            self.X_train = self.train.overview
            self.X_test = self.test.overview
   
        except:
            print('error ', (time.time()-start))
            return
        query = """cryptic messag bond past send trail uncov"""
        svc_score = {}
        self.SVC_pipeline = Pipeline([
          ('tfidf', TfidfVectorizer(stop_words=self.stopword_list)),
          ('clf', OneVsRestClassifier(LinearSVC() , n_jobs=1)),
        ])
        for category in categories:
          #print('... Processing {}'.format(category))
          # train the model using X_dtm & y
          self.SVC_pipeline.fit( self.X_train.values.astype('U'), self.train[category])
          # compute the testing accuracy
          prediction = self.SVC_pipeline.predict( self.X_test.values.astype('U'))
          svc_score[category] = self.SVC_pipeline.score( self.X_test.values.astype('U'), self.test[category])
          model = pickle.dumps(self.SVC_pipeline)
          self.models.append(model)
          
        # model scores average across categories
        svc_avg = 0.0
        n = len(categories)
        for i, category in enumerate(categories):
          svc_avg += svc_score[category]
          clf2 = pickle.loads(self.models[i])
          query_pred = clf2.predict([query])
          print('query prediction for {} is : {}'.format( category, query_pred) )    
          
        svc_avg = svc_avg / n
        print("SVC avg score: {:.2f}".format(svc_avg))
        
        print('sucess ', (time.time()-start))

    def process(self, df):
        doc = str(df['overview']).lower() # convert to lowercase
        title = str(df['title'])
        try:
            tokens = self.tokenizer.tokenize(doc)        # tokenize each overview
            tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stopword_list] # remove all stopwords/punctuation, and stem using porter stemmer
            self.term_freqs[title] = Counter(tokens)     # save the term frequency to list of all term frequencies
            self.doc_freqs += Counter(list(set(tokens))) # update document frequencies
            df['tokens'] = ' '.join(tokens)            
        except:
            print("error occured in process method")
            sys.exit()

        try:
            genre_str = str(df['genres'])
            genre_json = json.loads(genre_str)

            for i in genre_json:
                self.genre_freqs[i['name']] += 1
                self.genre_term_freqs[i['name']] += Counter(tokens)
                df[ i['name'] ] = 1
                
            return df
        except:
            print('error w/genre freq counter')
            sys.exit()
    
    def calcWeight(self, title, token):
        idf = self.getIDF(token)
        return (1+log10(self.term_freqs[title][token])) * idf

    def getIDF(self, token):
        if self.doc_freqs[token] == 0:
            return -1
        return log10(len(self.term_freqs)/self.doc_freqs[token])
    
    # calculate all of the tf-idf vectors 
    def calcTFIDF(self):
        for title in self.term_freqs:
            self.tf_idfs[title] = Counter()      # initialize the tf-idf vectors for each doc
            length = 0
            for token in self.term_freqs[title]:
                weight = self.calcWeight(title, token)
                self.tf_idfs[title][token] = weight
                length += weight**2
            self.doc_lens[title] = math.sqrt(length)
    
    # normalize all of the weights
    def normWeights(self):
        for title in self.tf_idfs:
            for token in self.tf_idfs[title]:
                self.tf_idfs[title][token] = self.tf_idfs[title][token] / self.doc_lens[title]
                if token not in self.postings_list:
                    self.postings_list[token] = Counter()
                self.postings_list[token][title] = self.tf_idfs[title][token]
    
    def getWeight(self, title, token):
        return self.tf_idfs[title][token]

    def find_film(self, name):
        try:
            results = self.query(name)
            films = []
            #df = pd.Series(np.random.randn( len(results[0]) ) ) # init a pandas series for the top 10 score values
            for i in range(len(results[0])):
                #print('name: {}, score: {}'.format(results[0][i], results[1][i]) )  
                temp_df = df_movies.loc[df_movies['title'].str.match(results[0][i])]
                temp_df['score'] = results[1][i]
                films.append(temp_df)     
            
            #print('\n')
            films = pd.concat(films)
            print('found film in df')
            films = films[['budget', 'title', 'id', 'overview', 'release_date', 'score', 'genres']]
            #print(films.head(10))
            row_count = df_movies.shape[0] # number of rows in dataframe
            if row_count < 10:
                films = films.head(row_count) 
            else:
                films = films.head(10)# save first 10 rows       
            films_dict = films.to_dict('records')
            return films_dict   
        except : 
            return None
    
    def query2(self, q_str):    
      tokens = self.tokenizer.tokenize(q_str)        # tokenize each overview
      tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stopword_list] # remove all stopwords/punctuation, and stem using porter stemmer
      return ' '.join(tokens)
        

    def query(self, q_str):
        q_str = q_str.lower() # lowercase the string
        q_tf = {} # query's term frequency 
        q_len = 0 # the query's euclidean lengths product
        docs = {}
        top_k = {} # store the upper bound
        cos_sims = Counter() # Counter for calc cosine sim between token & doc
        commondocs = None
        for token in q_str.split():
            token = self.stemmer.stem(token)
            print('token = ', token)
            if token not in self.postings_list:
                continue
            try:
                docs[token], weights = zip(*self.postings_list[token].most_common(10) )
            except:
                docs[token], weights = zip(*self.postings_list[token].most_common())

            top_k[token] = weights[9] # store the top-k corresponding elements in its postings list
            
            if commondocs is not None:
                commondocs = set(docs[token]) & commondocs # save only documents in both 
            else:
                commondocs = set(docs[token]) 
            
            q_tf[token] =  1 +log10(q_str.count(token))  # calculate the term frequencies of the query
            q_len += q_tf[token]**2
        q_len = sqrt(q_len)

        for doc in self.tf_idfs:
            cos_sim = 0
            for token in q_tf:
                if doc in docs[token]:
                    cos_sim = cos_sim + (q_tf[token] / q_len) * self.postings_list[token][doc]
                else:
                    cos_sim = cos_sim + (q_tf[token] / q_len) * top_k[token]

            cos_sims[doc] = cos_sim
        
        max_score = cos_sims.most_common(10)
        
        a,w = zip(*max_score)
        #print('a = ', a)
        return a, w

class Genre:
    def __init__(self):
        try:
            self.categories = set()
            start = time.time()
            df_movies.apply(self.process, axis=1)
            
        except:
            print('error')
        print('success', ( time.time() - start ) )
        print('genre list:', self.categories)
    
    def process(self, df):
        genre_str = str(df['genres'])
        genre_json = json.loads(genre_str)
        for i in genre_json:
            self.categories.add(i['name'])

class Genre2:
    def __init__(self, summary):
        # define cols needed for classification
        col = ['title','overview', 'genres']
        self.df = df_movies[col]
        self.df = self.df[pd.notnull(self.df['overview'])]
        self.df.columns = ['title','overview', 'genres']
        self.df['category_id'] = self.df['genres'].factorize()[0]

        # self.df.assign(mean_a=df.a.mean(), mean_b=df.b.mean())
        print(self.df.head())
        
        X_train, X_test, y_train, y_test = train_test_split(self.df['overview'], self.df['genres'], random_state = 0)
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(X_train)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        clf = MultinomialNB().fit(X_train_tfidf, y_train)

        print(clf.predict(count_vect.transform([summary])))

film_model = Film()

#film_model = Genre2('In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization.')


#film_model.query('hero')

# df = pd.read_csv('./tmdb_5000_movies.csv')
# print(find_film('superman'))
