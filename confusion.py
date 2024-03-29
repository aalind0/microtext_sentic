import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode

from sklearn.model_selection import KFold, cross_val_score
from sklearn import cross_validation

import pickle
import numpy as np

iv = open("iv.txt","r").read()
ovv = open("ovv.txt","r").read()

all_words = []
documents = []

allowed_word_types = ["J","R","V"]


for p in iv.split('\n'):
    documents.append( (p, "iv") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for p in ovv.split('\n'):
    documents.append( (p, "ovv") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())


save_documents = open("documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()

all_words = nltk.FreqDist(all_words)


word_features = list(all_words.keys())[:5000]


save_word_features = open("word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]


random.shuffle(featuresets)
print(len(featuresets))


testing_set = featuresets[3800:]
training_set = featuresets[:3800]

cv = cross_validation.KFold(len(training_set), n_folds=10, shuffle=True, random_state=None)


# for traincv, testcv in cv:
#     SGDC_classifier = SklearnClassifier(SGDClassifier())
#     clf = SGDC_classifier
#     SGDC_classifier.train(training_set[traincv[0]:traincv[len(traincv)-1]])
#     print('accuracy:', nltk.classify.util.accuracy(SGDC_classifier, training_set[testcv[0]:testcv[len(testcv)-1]]))

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
for traincv, testcv in cv:
    SGDC_classifier = SklearnClassifier(SGDClassifier())
    clf = SGDC_classifier
    y_pred = cross_val_predict(SGDC_classifier, training_set[testcv[0]:testcv[len(testcv)-1]])
    y = traincv
    conf_mat = confusion_matrix(y,y_pred)
