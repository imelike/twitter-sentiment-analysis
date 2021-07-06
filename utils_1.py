import re
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

## PREPROCESSING(ON ISLEME) fonksiyonu ##
def process_tweet(tweet): 
    
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean


## FREKANS DICTIONARY OLUSTURMA ##
def build_freqs(tweets, ys):
    # Boş bir sözlükle başlayıp tüm tweet'leri döngüye alarak, her tweet için içerisindeki kelimeleri,
    # işlenmiş halleri üzerinden frekans listesine ekliyoruz.
  
    # y lerin olduğu Numpy array python listesine çevrilir.
    # Eğer y bir numpy array se y'nin boyutu m*1 old. icin:
    # np.squeeze ile array veri kaybı olmadan m e dönüştürülür.
    # to list ile de m boyutunda np arrayden python listesine çevrilir.
    yslist = np.squeeze(ys).tolist()

    # Boş bir frekans listesiyle başlayıp y ler ve tweetler için:
    # ylist'tekileri y'ye tweets listesindekileri word'e atıyor.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        
        for word in process_tweet(tweet): #token preprocessing olur.
            pair = (word, y) #pair = kelime+kelimenin etiketidir, type ı tuple olur.
            if pair in freqs: #pair freqs(frekans) listesinin içerisindeyse degerini 1 artılır. yoksa da ilk kez eklenir.
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs