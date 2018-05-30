import numpy, pyhacrf, nltk
from pyhacrf import StringPairFeatureExtractor, Hacrf

training_X = []

iv = open("iv.txt","r").read()
ovv = open("ovv.txt","r").read()

for p in iv.split("\n"):
    for q in ovv.split("\n"):
        training_X.append((p, q))
        #print(training_X)
    #training_X.append(('p', 'q'))
    #print(training_X)

training_y = ['match',
              'match',
              'match',
              'non-match',
              'non-match']

# Extract features
feature_extractor = StringPairFeatureExtractor(match=True, numeric=True)
training_X_extracted = feature_extractor.fit_transform(training_X)

# Train model
model = Hacrf(l2_regularization=1.0)
model.fit(training_X_extracted, training_y)

# Evaluate
from sklearn.metrics import confusion_matrix
predictions = model.predict(training_X_extracted)
