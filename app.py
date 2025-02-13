#Data
import pandas as pd
import numpy as np

#NLP
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

#Pre-Processing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#Modeling
from sklearn.svm import SVC
from sklearn.metrics import f1_score

df = pd.read_csv('spam.csv', encoding='latin-1')
#df.head()
df.drop([df.columns[col] for col in [2, 3, 4]], axis=1, inplace=True)
encoder = LabelEncoder()
df['v1'] = encoder.fit_transform(df['v1'])
class_mappings = {index: label for index, label in enumerate(encoder.classes_)}

# Take an email string and convert it to a list of stemmed words
def processEmail(mailcontents):
    ps = PorterStemmer()
    mailcontents = re.sub(r'<[^<>]+>',' ', mailcontents)
    mailcontents = re.sub(r'[0-9]+', 'mumber' ,mailcontents)
    mailcontents = re.sub(r'(http|https)://[^\s]*' , 'httpaddr',mailcontents)
    mailcontents = re.sub(r'[^\s]+@[^\s]+', 'mailaddr',mailcontents)
    mailcontents = re.sub(r'[$]+', 'dollar',mailcontents)

    words = word_tokenize(mailcontents)
    for i in range(len(words)):
        words[i] = re.sub(r'a-zA-Z0-9', '', words[i])
        words[i] = ps.stem(words[i])
    words = [word for word in words if len(word)>=1] 
    return words

# Take a list of emails and get a dictionary of the most common words
def getVocabulary(mails, vocab_length):
    vocabulary = dict()
    for i in range(len(mails)):
        mails[i] = processEmail(mails[i])
        for word in mails[i]:
            if word in vocabulary.keys():
                vocabulary[word] += 1

            else:
                vocabulary[word] = 1
    vocabulary = sorted(vocabulary.items(), key=lambda x: x[1], reverse=True)
    vocabulary = list(map(lambda x: x[0], vocabulary[0:vocab_length]))
    vocabulary = {index: word for index, word in enumerate(vocabulary)}
    
    return vocabulary
# Get a dictionary key given a value
def getKey(dictionary, val):
    for key, value in dictionary.items():
        if value == val:
            return key
        
# Get the indices of vocab words used in a given email
def getIndices(email, vocabulary):
    word_indices = set()
    
    for word in email:
        if word in vocabulary.values():
            word_indices.add(getKey(vocabulary, word))
    
    return word_indices
def getFeatureVector(word_indices, vocab_length):
    feature_vec = np.zeros(vocab_length)
    
    for i in word_indices:
        feature_vec[i] = 1
        
    return feature_vec
vocab_length = 2000

vocabulary = getVocabulary(df['v2'].to_list(), vocab_length)

emails = df['v2'].to_list()
emails = list(map(lambda x: processEmail(x), emails))    

X = list(map(lambda x: getFeatureVector(getIndices(x, vocabulary), vocab_length), emails))
X = pd.DataFrame(np.array(X).astype(np.int16))

y = df['v1']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
model = SVC()

model.fit(X_train, y_train)
model.score(X_test, y_test)
y_pred = model.predict(X_test)
f1_score(y_test, y_pred)