import re
import numpy as np

import glove
from nltk.corpus import gutenberg
from multiprocessing import Pool
from scipy import spatial

sentences = list('nus_sms-data.txt')
