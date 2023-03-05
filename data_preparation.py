
#importing necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from numpy.linalg import norm
import re
import warnings
warnings.filterwarnings("ignore")

def get_data_preparation():
    """
    Function to prepare the data: selecting title and text
    """
    data=pd.read_csv(r"fake_dataset.csv")
    data_select=data[['title','text']]
    data_select.dropna(inplace=True)
    return data_select

def cosine_sim(row):
    """
    define a function to compute cosine similarity row-wise
    """
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(row.values)
    cos_sim = cosine_similarity(tfidf)
    return cos_sim[0][1]

def get_cosine_similarity():
    """
    Function to call cosine_sim and get_data_preparation to get cosine similarith
    """
    # apply the function to each row of the DataFrame
    data_select=get_data_preparation()
    data_select['cosine_sim'] = data_select[['title', 'text']].apply(cosine_sim, axis=1)
    
    #renaming the columns
    data_select.columns=['text_a','text_b','label']
    return data_select