import json, nltk, math # data format, text processing, calculations
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import string
import time
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from math import log10, sqrt
from collections import Counter

df_movies = pd.read_csv('./tmdb_5000_movies.csv')
class Film:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
        self.tf_idfs = {}                        # tf-idf vectors
        self.term_freqs = {}                     # (tfs) term frequencies of all tokens in all documents
        self.doc_freqs = Counter()               # document frequencies
        self.doc_lens = Counter()               # document lengths
        self.postings_list = {}                 # a sorted list in which each element is in the form (doc d, tf-idf weight w)
        self.categories = { 'Western', 'Mystery', 'Foreign', 'History', 'Thriller', 'TV Movie', 'Drama', 
            'Horror', 'Adventure', 'Documentary', 'Crime', 'Comedy', 'Family', 'War', 'Romance', 
            'Fantasy', 'Animation', 'Action', 'Music', 'Science Fiction' }
        self.stopword_list = set( stopwords.words('english') ).union( list(string.punctuation) )

        start = time.time()
        try:
            df_movies.apply(self.process, axis=1) # perform additonal processing
            self.calcTFIDF()
            self.normWeights()
        except:
            print('error ', (time.time()-start))
        
        print('sucess ', (time.time()-start))

           
    def process(self, df):
        doc = str(df['overview']).lower() # convert to lowercase
        title = str(df['title'])
        try:
            tokens = self.tokenizer.tokenize(doc)        # tokenize each overview
            tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stopword_list] # remove all stopwords/punctuation, and stem using porter stemmer
            self.term_freqs[title] = Counter(tokens)   # save the term frequency to list of all term frequencies
            self.doc_freqs += Counter(list(set(tokens))) # update document frequencies           
        except:
            print("error occured in process method")
            return None
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
            films = films[['budget', 'title', 'id', 'overview', 'release_date', 'score']]
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


film_model = Genre()

#film_model.query('hero')

# df = pd.read_csv('./tmdb_5000_movies.csv')
# print(find_film('superman'))
