import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

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
    else:
        documents_ovv.append((p, "ovv"))
    i+=1

print(documents_ovv)
