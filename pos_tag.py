import nltk,re
import json
from nltk.tokenize import word_tokenize

iv = open("another_iv.txt","r").read()
ovv = open("another_ovv.txt","r").read()


all_words = []
documents = []

allowed_word_types = ["J","R","V"]


for p in iv.split('\n'):
    twitter_username_re = re.sub(r'@([A-Za-z0-9_]+)','', p)
    twitter_username_re = re.sub(r'http\S+', '', twitter_username_re)
    words = word_tokenize(twitter_username_re)
    pos = nltk.pos_tag(words)
    #print(pos)



for p in ovv.split('\n'):
    twitter_username_re = re.sub(r'@([A-Za-z0-9_]+)','', p)
    twitter_username_re = re.sub(r'http\S+', '', twitter_username_re)
    print(twitter_username_re)
    words = word_tokenize(twitter_username_re)
    pos = nltk.pos_tag(words)
    #print(pos)




    #with open("data.txt", "w") as output:
            #output.write(str(pos))
