import gzip
import gensim
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

data_file = "nus_sms-data.txt.gz"

with gzip.open ('nus_sms-data.txt.gz', 'rb') as f:
    for i,line in enumerate (f):
        print(line)

def read_input(input_file):

    logging.info("reading file {0}...this may take a while".format(input_file))

    with gzip.open (input_file, 'rb') as f:
        for i, line in enumerate (f):

            if (i%100==0):
                logging.info ("read {0} reviews".format (i))
            # do some pre-processing and return a list of words for each review text
            yield gensim.utils.simple_preprocess (line)


documents = list (read_input (data_file))
logging.info ("Done reading data file")

model = gensim.models.Word2Vec (documents, size=10, window=10, min_count=2, workers=10)
model.train(documents,total_examples=len(documents),epochs=50)

w1 = "are"
print(model.wv.most_similar (positive=w1, topn=5))

print(model.wv.similarity(w1="you",w2="u"))
