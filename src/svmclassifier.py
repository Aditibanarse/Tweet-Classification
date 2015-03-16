'''
Created on Nov 10, 2013

@author: Deepan
'''

import re
import nltk
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics.metrics import accuracy_score

# initialize stopWords
stopWords = []

# start process_tweet
def processTweet(tweet):
    # process the tweets
 
    # Convert to lower case
    tweet = tweet.lower()
    # Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[\s]+)|(https?://[^\s]+))', 'URL', tweet)
    # Convert @username to AT_USER
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet)
    # Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    # Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    # trim
    tweet = tweet.strip('\'"')
    return tweet
# end

# start replaceTwoOrMore
def replaceTwoOrMore(s):
    # look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
# end
 
# start getStopWordList
def getStopWordList(stopWordListFileName):
    # read the stopwords file and build a list
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')
 
    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
# end
 
# start getfeatureVector
def getFeatureVector(tweet):
    featureVector = []
    # split tweet into words
    words = tweet.split()
    for w in words:
        # replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        # strip punctuation
        w = w.strip('\'"?,.')
        # check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        # ignore if it is a stop word
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector
# end

def getSVMFeatureVectorAndLabels(tweets, featureList):
    sortedFeatures = sorted(featureList)
    map = {}
    feature_vector = []
    labels = []
    for t in tweets:
        label = 0
        map = {}
        # Initialize empty map
        for w in sortedFeatures:
            map[w] = 0
        
        tweet_words = t[0]
        tweet_opinion = t[1]
        # Fill the map
        for word in tweet_words:
            # process the word (remove repetitions and punctuations)
            word = replaceTwoOrMore(word) 
            word = word.strip('\'"?,.')
            # set map[word] to 1 if word exists
            if word in map:
                map[word] = 1
        # end for loop
        values = map.values()
        feature_vector.append(values)
        if(tweet_opinion == '0'):
            label = 0
        elif(tweet_opinion == '1'):
            label = 1
        labels.append(label)            
    # return the list of feature_vector and labels
    return {'feature_vector' : feature_vector, 'labels': labels}
# end
 
def union(a, b):
    return list(set(a) | set(b))

st = open('../data/stopwords.txt', 'r')
stopWords = getStopWordList('../data/stopwords.txt')
 
fp = open('../data/smokingtweets.txt', 'r')
line = fp.readline()

tweets = []
featureList = []

while line:
    lineSplit = line.rstrip().split('|@~')
    tweet = lineSplit[0]
    sentiment = lineSplit[1]
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet)
    tweets.append((featureVector, sentiment))
    featureList = union(featureList, featureVector)
    line = fp.readline()
# end loop

tp = open('../data/testtweets.txt', 'r')
tLine = tp.readline()

testTweets = []

while tLine:
    lines = tLine.rstrip().split('|@~')
    tweet = lines[0]
    sentiment = lines[1]
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet)
    testTweets.append((featureVector, sentiment))
    tLine = tp.readline()
# end loop

# Train the SVM Classifier
result_train = getSVMFeatureVectorAndLabels(tweets, featureList)
result_test = getSVMFeatureVectorAndLabels(testTweets, featureList)
# Split the data into a training set and a test set
data_train = result_train['feature_vector']
target_train = result_train['labels']
data_test = result_test['feature_vector']
target_test = result_test['labels']

# Run SVM Classifier
SVMClassifier = svm.SVC(kernel='linear')

target_pred = SVMClassifier.fit(data_train, target_train).predict(data_test)


targetNames = ['cessation', 'no cessation']
print "Classification by SVM Classifier"
print classification_report(target_test, target_pred, target_names=targetNames)
print confusion_matrix(target_test, target_pred)
print accuracy_score(target_test, target_pred)

#

