import numpy as np
import re
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords

movie_data = load_files(r"/home/admini/Aalind/microtext_sentic/metrics/test2/review_polarity/txt_sentoken")
X, y = movie_data.data, movie_data.target

documents = []

from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))

    # # remove all single characters
    # document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    #
    # # Remove single characters from the start
    # document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # # Removing prefixed 'b'
    # document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    # document = document.lower()
    #
    # # Lemmatization
    # document = document.split()
    #
    # document = [stemmer.lemmatize(word) for word in document]
    # document = ' '.join(document)

    documents.append(document)
print(documents)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer(max_features=4000)
X = tfidfconverter.fit_transform(documents).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestRegressor
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(X_train)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, y_train))*100)

# classifier = RandomForestRegressor(n_estimators=1000, random_state=0)
# classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


# documents = []
#
# from nltk.stem import WordNetLemmatizer
#
# stemmer = WordNetLemmatizer()
#
# for sen in range(0, len(X)):
#     # Remove all the special characters
#     document = re.sub(r'\W', ' ', str(X[sen]))
#
#     # Substituting multiple spaces with single space
#     document = re.sub(r'\s+', ' ', document, flags=re.I)
#
#     # Converting to Lowercase
#     document = document.lower()
#
#     # # Lemmatization
#     # document = document.split()
#     #
#     # document = [stemmer.lemmatize(word) for word in document]
#     # document = ' '.join(document)
#
#     documents.append(document)
#
# # print(documents)
#
# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer(max_features=4000)
# X = vectorizer.fit_transform(documents).toarray()
#
# from sklearn.feature_extraction.text import TfidfTransformer
# tfidfconverter = TfidfTransformer()
# X = tfidfconverter.fit_transform(X).toarray()
#
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
# from sklearn.ensemble import RandomForestRegressor
#
# classifier = RandomForestRegressor(n_estimators=1000, random_state=0)
# classifier.fit(X_train, y_train)
#
# y_pred = classifier.predict(X_test)
#
# # Evaluating the model
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))
# print(accuracy_score(y_test, y_pred))
