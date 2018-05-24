import re
import numpy as np

from gensim.models.word2vec import Text8Corpus
import glove
from multiprocessing import Pool
from scipy import spatial
import itertools

sentences = list(itertools.islice(Text8Corpus('text8'),None))
iv = open("iv.txt","r").read()

iv.fit(sentences, window = 10)
