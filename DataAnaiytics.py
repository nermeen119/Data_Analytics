import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from nltk.cluster import KMeansClusterer
import nltk
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer
from textblob.classifiers import NaiveBayesClassifier
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from numpy import random, where
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
input_str=pd.read_csv("E:/Data Analytics/Labs/googleplaystore_user_reviews.csv")
input_str['Translated_Review'].fillna("No chars in this box",inplace=True)
input_str = input_str['Translated_Review']
lenght= len(input_str.head(20))
#[Sentiment]
stemmer= PorterStemmer()
input_str=pd.read_csv("E:/Data Analytics/googleplaystore_user_reviews.csv")
for word in input_str:
 s=SentimentIntensityAnalyzer()
 vs = s.polarity_scores(word)
 print(word,str(vs))


#[Stemming]
stemmer= PorterStemmer()
i=0
while i<lenght:
 for word in word_tokenize(input_str[i]):
  print(stemmer.stem(word))
 i += 1
 print("--------------------------------------------------------")

#[Lemmatization]
lemmatizer=WordNetLemmatizer()
j=0
while j<lenght:
 for word in word_tokenize(input_str[j]):
  print(lemmatizer.lemmatize(word))
 j += 1
 print("--------------------------------------------------------")

# [stop_words]
stop_words = set(stopwords.words('english'))
print(stop_words)
print('**************************************************************************************************************')
ii=0
while ii<lenght:
 tokens = word_tokenize(input_str[ii])
 result = [i for i in tokens if not i in stop_words]
 print (result)
 ii += 1

#[Lowering]
index=0
while index<lenght:
 LowerChar = input_str[index].lower()
 print(LowerChar)
 index+=1

#[Remove puntuation]
import string
index1=0
while index1<lenght:
 result = input_str[index1].translate(str.maketrans('','', string.punctuation))
 print(result)
 index1+=1
 print("--------------------------------------------------------")



data = pd.read_csv("E:/Data Analytics/ceds_music.csv")
DataScan=data.head(1000)
x= DataScan['len']
x=x.values.reshape(-1,2)
lof=LocalOutlierFactor(n_neighbors=4,contamination=.03)
y_pred=lof.fit_predict(x)
print(y_pred)
lofs_index=where(y_pred==-1)
values=x[lofs_index]
print(lofs_index)
plt.scatter(x[:,0], x[:,1])
plt.scatter(values[:,0], values[:,1], color='r')
plt.show()

