import pandas as pd
from keras.models import load_model
import pickle
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA

pd.options.display.max_seq_items = None

X_train = pd.read_csv('data/X_train_with_division_non_unique_words.csv', index_col=0)
y_train = pd.read_csv('data/y_train.csv', index_col=0)
test_df = pd.read_csv('data/test_df_with_division_non_unique_words.csv', index_col=0)

print(X_train.columns)

X_train = X_train.iloc[:,22:]
test_df = test_df.iloc[:,22:]

print(X_train.columns)

print("Data loaded")
print(X_train.shape)
print(y_train.shape)

def training():
    classifier = AdaBoostClassifier(RandomForestClassifier())

    classifier.fit(X_train, y_train)

    print(f1_score(classifier.predict(X_train), y_train, average='macro'))

    filename = 'models_test/adaboost_model_example.sav'
    pickle.dump(classifier, open(filename, 'wb'))

training()

def prediction():

    classifier = pickle.load(open('models_test/adaboost_model_example.sav', 'rb'))

    preds = classifier.predict(test_df)

    pred_df = pd.DataFrame(preds, columns=['label'])

    pred_df.to_csv("predictions_test/adaboost_example.csv", index=True, index_label='Id')

prediction()
