import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pickle

from keras.models import Sequential, Model,load_model

X_train = pd.read_csv('../data/selected_features_train.csv', index_col=0)
y_train = pd.read_csv('../data/y_train.csv', index_col=0)
test_df = pd.read_csv('../data/selected_features_test.csv', index_col=0)

# X_train = pd.read_csv('../data/X_train_with_division_non_unique_words.csv', index_col=0)
# y_train = pd.read_csv('../data/y_train.csv', index_col=0)
# test_df = pd.read_csv('../data/test_df_with_division_non_unique_words.csv', index_col=0)

X_train = X_train.iloc[:,:]
test_df = test_df.iloc[:,:]

print("Data loaded")
print(X_train.shape)
print(y_train.shape)
print(test_df.shape)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

pca = PCA(100)

pca.fit(X_train,y_train)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
test_df = pca.transform(test_df)

classifier = KNeighborsClassifier(n_neighbors=15, p=1)

classifier.fit(X_train, y_train)
# scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring='f1_macro')

print(f1_score(classifier.predict(X_train), y_train, average='macro'))
print(f1_score(classifier.predict(X_test), y_test, average='macro'))


