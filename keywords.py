import nltk,re
import json
from nltk.tokenize import word_tokenize

pos = open("positive.txt","r").read()
neg = open("negative.txt","r").read()

file = open("dataset_test.txt", "w")

str = "a little"

for p in pos.split("\n"):
    if str in p:
        file.write(p+"\n")

file.close()
