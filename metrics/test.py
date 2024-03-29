#from nltk.metrics.scores import (precision, recall)

import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode

from nltk.metrics.scores import (precision, recall)

import pickle
import numpy as np

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

iv = open("iv.txt","r").read()
ovv = open("ovv.txt","r").read()

all_words = []
documents = []

allowed_word_types = ["J","R","V",
"CC",
"CD",
"DT",
"EX",
"FW",
"IN",
"JJ",
"JJR",
"JJS",
"LS",
"MD",
"NN",
"NNS",
"NNP",
"NNPS",
"PDT",
"POS",
"PRP",
"PRP$",
"RB",
"RBR",
"RBS",
"RP",
"TO",
"UH",
"VB",
"VBD",
"VBG",
"VBN",
"VBP",
"VBZ",
"WDT",
"WP",
"WP$",
"WRB"]


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


save_features = open("featuresets.pickle","wb")
pickle.dump(featuresets, save_features)
save_features.close()

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(NuSVC_classifier, testing_set))
print(classification_report(NuSVC_classifier, testing_set))
print(accuracy_score(NuSVC_classifier, testing_set))

# LinearSVC_classifier = SklearnClassifier(LinearSVC())
# LinearSVC_classifier.train(training_set)
# print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
