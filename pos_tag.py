import nltk
import json
from nltk.tokenize import word_tokenize

iv = open("iv.txt","r").read()
ovv = open("ovv.txt","r").read()

file2 = open("pos_iv.txt", "w")

all_words = []
documents = []

allowed_word_types = ["J","R","V"]


for p in iv.split('\n'):
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)



for p in ovv.split('\n'):
    documents.append( (p, "ovv") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    print(pos)



    #with open("data.txt", "w") as output:
            #output.write(str(pos))
