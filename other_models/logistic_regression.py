import numpy as np
import pandas as pd
import pickle

pd.options.display.max_seq_items = None

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from keras.models import Sequential, Model, load_model
from sklearn.decomposition import PCA

from sklearn.metrics import f1_score


X_train = pd.read_csv('../data/selected_features_train.csv', index_col=0)
y_train = pd.read_csv('../data/y_train.csv', index_col=0)
test_df = pd.read_csv('../data/selected_features_test.csv', index_col=0)

X_train = X_train.iloc[:,:]
test_df = test_df.iloc[:,:]

print("Data loaded")
print(X_train.shape)
print(y_train.shape)
print(test_df.shape)

# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

classifier = LogisticRegression(penalty='l2', multi_class='ovr')

classifier.fit(X_train, y_train)
# scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring='f1_macro')

# print(f1_score(classifier.predict(X_train), y_train, average='macro'))
# print(f1_score(classifier.predict(X_test), y_test, average='macro'))

# print ("mean validation F1 for Random Forests:",
#        "%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

filename = 'logistic_regression_best_features.sav'
pickle.dump(classifier, open(filename, 'wb'))
