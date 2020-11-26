import numpy as np 
import pandas as pd
from flask import Flask, request, jsonify, render_template 
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords
import string
import re
from textstat.textstat import *

def textclean(test_string):
    #cleaning the data
    username_pattern = "@[\w\-\:]+"
    retweet_pattern = "RT+"
    space_pattern = "\s+"
    url_pattern = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    miscellaneous_pattern = "&#[0-9\;]+"
    ampersand_pattern = "&amp"
    result = re.sub(username_pattern,' ',test_string)
    result = re.sub(retweet_pattern,' ',result)
    result = re.sub(miscellaneous_pattern,' ',result)
    result = re.sub(url_pattern,' ',result)
    result = re.sub(ampersand_pattern,'and',result)
    result = re.sub(space_pattern,' ',result)
    return result

def textTokenize(tweet):
    #tokenize and stemming
    stemmer = PorterStemmer()
    test_string = " ".join(re.split("[^a-zA-Z1-9]+", tweet.lower())).strip()
    #print(test_string)
    tokens = [stemmer.stem(t) for t in test_string.split()]
    #print(tokens)
    return tokens

def basicTextTokenize(tweet):
    """Just tokenize and not use stemmer here"""
    token_tweet = " ".join(re.split("[^a-zA-Z1-9]+",tweet.lower())).strip()
    #print(token_tweet.split())
    return token_tweet.split()

def pos_tagging(tokens):
    pos_tweets = nltk.pos_tag(tokens)
    tagged_tweets = [x[1] for x in pos_tweets]
    return tagged_tweets

sentiment_analyzer = VS()

def countTwitterObjects(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    4) hashtags with HASHTAGHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned.
    
    Returns counts of urls, mentions, and hashtags.
    """
    space_pattern = "\s+"
    url_pattern = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    username_pattern = "@[\w\-\:]+"
    hashtag_pattern = "#[\w\-]+"
    parsed_text = re.sub(space_pattern,' ',text_string)
    parsed_text = re.sub(url_pattern, 'URLHERE',parsed_text)
    parsed_text = re.sub(username_pattern,'MENTIONHERE',parsed_text)
    parsed_text = re.sub(hashtag_pattern,'HASHTAGHERE',parsed_text)
    return (parsed_text.count('URLHERE'),parsed_text.count('MENTIONHERE'),parsed_text.count('HASHTAGHERE'))

def otherFeatures(tweet):
    """
    This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features
    """
    sentiment = sentiment_analyzer.polarity_scores(tweet)
    words = textclean(tweet) 
    
    syllables = textstat.syllable_count(words)
    num_chars = sum(len(w) for w in words)
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables+0.001))/float(num_words+0.001),4)
    num_unique_terms = len(set(words.split()))
    
    ###Modified FK grade, where avg words per sentence is just num words/1
    FKRA = round(float(0.39 * float(num_words)/1.0) + float(11.8 * avg_syl) - 15.59,1)
    ##Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015*(float(num_words)/1.0) - (84.6*float(avg_syl)),2)
    
    twitter_objs = countTwitterObjects(tweet)
    retweet = 0
    if "rt" in words:
        retweet=1
    
    features = [FKRA, FRE,syllables, avg_syl, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'],
                twitter_objs[2], twitter_objs[1],
                twitter_objs[0], retweet]
    return features

def getFeatures(tweets):
    feats = []
    for t in tweets:
        feats.append(otherFeatures(t))
    print(np.array(feats))
    return np.array(feats)

stopwords = stopwords = nltk.corpus.stopwords.words("english")
other_exclusions = ["#ff","ff","rt"]
stopwords.extend(other_exclusions)

app = Flask(__name__)
pre_trained_model = pickle.load(open('Logistic_R_1_5ngrams.pkl','rb'))
word_tfidf = pickle.load(open('wordtfidf.pkl','rb'))
pos_tag_tfidf = pickle.load(open('postagtfidf.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
    text_string = str(request.form['text'])
    #print(type(text_string))
    print(text_string)
    temp_clean_text = basicTextTokenize(textclean(text_string))
    print(temp_clean_text)
    temp_tweet_tags = []
    temp_pos_tags = pos_tagging(temp_clean_text)
    temp_tag_str = " ".join(temp_pos_tags)
    temp_tweet_tags.append(temp_tag_str)
    print(temp_tweet_tags)
    s_list = []
    s_list.append(text_string)
    vectorizer_new = TfidfVectorizer(
    tokenizer = textTokenize,
    preprocessor = textclean,
    ngram_range = (1,5),
    stop_words = stopwords,
    use_idf = True,
    smooth_idf = False,
    norm = None,
    decode_error = 'replace',
    max_features = 10000,
    vocabulary = word_tfidf,
    min_df = 5,
    max_df = 0.75
    )
    pos_vectorizer_new = TfidfVectorizer(
    tokenizer = None,
    lowercase = False,
    preprocessor = None,
    ngram_range = (1,5),
    stop_words = None,
    use_idf = False,
    smooth_idf = False,
    norm = None,
    decode_error = 'replace',
    max_features = 5000,
    vocabulary = pos_tag_tfidf,
    min_df = 5,
    max_df = 0.75,
    )
    new_temp_tfidf = vectorizer_new.fit_transform(s_list).toarray()
    temp_pos = pos_vectorizer_new.fit_transform(pd.Series(temp_tweet_tags)).toarray()
    temp_feats = getFeatures(s_list)
    print(temp_feats)
    temp_M = np.concatenate([new_temp_tfidf,temp_pos,temp_feats],axis=1)
    temp_X = pd.DataFrame(temp_M)
    y_pred = pre_trained_model.predict(temp_X)
    
    value = y_pred.item()
    print(type(value))
    print(value)
    if(value == 0):
        output = "Hate text"
        print("hate SPeech")
    elif(value == 1):
        output = "Offensive text"
        print("Offensive")
    else:
        output = "Normal text"
        print("normal")

    return render_template('index.html',default_html='You entered-\n',input_string=text_string,predict_text='which is {}'.format(output))

if __name__=="__main__":
    app.run(debug=True)
