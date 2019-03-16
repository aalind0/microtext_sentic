from sklearn.feature_extraction.text import CountVectorizer
from nltk.classify.scikitlearn import SklearnClassifier

from nltk.classify import ClassifierI
from statistics import mode

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

data = []
data_labels = []

with open("pre_iv.txt") as f:
    for i in f:
        data.append(i)
        data_labels.append('iv')

with open("pre_ovv.txt") as f:
    for i in f:
        data.append(i)
        data_labels.append('oov')

vectorizer = CountVectorizer(
    analyzer = 'char',
    lowercase = False,
    ngram_range=(1, 3)
)
features = vectorizer.fit_transform(
    data
)
features_nd = features.toarray() # for easy usage

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test  = train_test_split(
        features_nd,
        data_labels,
        train_size=0.80,
        random_state=1234)

# Logistic Regression
print("\n" + "Logistic regression")
from sklearn.linear_model import LogisticRegression, SGDClassifier
log_model = LogisticRegression()

log_model = log_model.fit(X=X_train, y=y_train)

y_pred = log_model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

print("\n" + "SGDC_classifier")
log_model_SGDC = SGDClassifier()

log_model_SGDC = log_model_SGDC.fit(X=X_train, y=y_train)

y_pred = log_model_SGDC.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


# NuSVC Classifier
print("\n" + "NuSVC")
from sklearn.svm import SVC, LinearSVC, NuSVC
log_model1 = NuSVC()
log_model1 = log_model1.fit(X=X_train, y=y_train)
y_pred = log_model1.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

# LinearSVC_classifier
print("\n" + "LinearSVC_classifier")
log_model2 = LinearSVC()
log_model2 = log_model2.fit(X=X_train, y=y_train)
y_pred = log_model2.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

# SVC_classifier
print("\n" + "SVC_classifier")
log_model3 = LinearSVC()
log_model3 = log_model3.fit(X=X_train, y=y_train)
y_pred = log_model3.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

from sklearn.naive_bayes import MultinomialNB, BernoulliNB

# MultinomialNB_classifier
print("\n" + "MultinomialNB")
log_model_multinomial = MultinomialNB()
log_model_multinomial = log_model_multinomial.fit(X=X_train, y=y_train)
y_pred = log_model_multinomial.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

# BernoulliNB ClassifierI
print("\n" + "BernoulliNB")
log_model_bernoulli = BernoulliNB()
log_model_bernoulli = log_model_bernoulli.fit(X=X_train, y=y_train)
y_pred = log_model_bernoulli.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
