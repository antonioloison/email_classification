import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pickle

from keras.models import Sequential, Model,load_model

X_train = pd.read_csv('data/selected_features_train.csv', index_col=0)
y_train = pd.read_csv('data/y_train.csv', index_col=0)
test_df = pd.read_csv('data/selected_features_test.csv', index_col=0)

X_train = X_train.iloc[:,:]
test_df = test_df.iloc[:,:]

print("Data loaded")
print(X_train.shape)
print(y_train.shape)
print(test_df.shape)

def training():
    classifier = SVC(C=10000)

    classifier.fit(X_train, y_train)

    print(f1_score(classifier.predict(X_train), y_train, average='macro'))

    filename = 'models_test/support_vector_machine.sav'
    pickle.dump(classifier, open(filename, 'wb'))

training()

def prediction():

    classifier = pickle.load(open('models_test/support_vector_machine.sav', 'rb'))

    preds = classifier.predict(test_df)

    pred_df = pd.DataFrame(preds, columns=['label'])

    pred_df.to_csv("predictions_test/support_vector_machine.csv", index=True, index_label='Id')

prediction()
