# importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import re
#nltk.data.path.append('./nltk_data/')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pickle


## ---------------------------------
##Read Data 
## ---------------------------------
    
df = pd.read_csv("twitter_sentiments.csv")

# Delete Columns
del df["SentimentSource"]
del df["Unnamed: 4"]
del df["Unnamed: 5"]
del df["Unnamed: 6"]
del df["ItemID"]


## ---------------------------------
##Data Preprocessing 
# ---------------------------------
# convert to lower case
df['clean_tweet'] = df['SentimentText'].str.lower()
# Remove punctuations
df['clean_tweet'] = df['clean_tweet'].str.replace('[^\w\s]',' ')
# Remove spaces in between words
df['clean_tweet'] = df['clean_tweet'].str.replace(' +', ' ')
# Remove Numbers
df['clean_tweet'] = df['clean_tweet'].str.replace('\d+', '')
# Remove trailing spaces
df['clean_tweet'] = df['clean_tweet'].str.strip()
# Remove URLS
df['clean_tweet'] = df['clean_tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
# remove stop words
stop = stopwords.words('english')
stop.extend(["racism","alllivesmatter","amp","https","co","like","people","black","white"])
df['clean_tweet'] =  df['clean_tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop ))



## ---------------------------------
##Prepare Random Forest Model
##Define Pipeline Stages
## ---------------------------------

tfidf_vectorizer = TfidfVectorizer(lowercase= True, max_features=1000, stop_words=ENGLISH_STOP_WORDS)
tfidf_vectorizer.fit(df.clean_tweet)

# transform the train and test data,
tweets_idf = tfidf_vectorizer.transform(df.clean_tweet)


# create the object of Random Forest Model
model_RF = RandomForestClassifier(n_estimators=100)
model_RF.fit(tweets_idf, df.Sentiment)

pipeline = Pipeline(steps= [('tfidf', TfidfVectorizer(lowercase=True,
                                                      max_features=1000,
                                                      stop_words= ENGLISH_STOP_WORDS)),
                                                      ('model', RandomForestClassifier(n_estimators = 100))])

# fit the pipeline model with the training data                            
pipeline.fit(df.clean_tweet, df.Sentiment)
    
# save the model
# import joblib
from joblib import dump

# dump the pipeline model
dump(pipeline, filename="text_classification.joblib", compress= True)

    

        
        
        
        