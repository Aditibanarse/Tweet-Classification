import re
import nltk

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
 
def union(a, b):
    return list(set(a) | set(b))

# start extract_features
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features
# end

st = open('../data/stopwords.txt', 'r')
stopWords = getStopWordList('../data/stopwords.txt')
 
# Read the tweets one by one and process it
# inpTweets = csv.reader(open('../data/sampletweets.csv', 'rb'), delimiter=',', quotechar='|')
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
    featureList = union(featureList, featureVector)
    tweets.append((featureVector, sentiment))
    line = fp.readline()
# end loop

# print featureList
training_set = nltk.classify.util.apply_features(extract_features, tweets)

# Train the classifier
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)

# Test the classifier
testTweet = 'I chew on my e-cig like a pen... No emphysema but I\'ll have messed up teef... Smh'
processedTestTweet = processTweet(testTweet)
print NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet)))
