import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
from sklearn import svm
import pickle
import numpy as np


documents = []

documents_ovv = []

i = 1

short_line = open("nus_sms-data.txt","r").read()

#for p in short_line.readlines():
    #if i%2 == 0:
        #print(line)

for p in short_line.split('\n'):
    if i%2 == 0:
        documents.append((p, "iv"))
        words = word_tokenize(p)

    else:
        documents.append((p, "ovv"))
        words = word_tokenize(p)
    i+=1

print(documents)
#save_documents = open("documents.pickle", "wb")
#pickle.dump(documents, save_documents)
#save_documents.close()
