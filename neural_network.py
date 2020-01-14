import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization
pd.options.display.max_seq_items = None


# Read csvs

X_train = pd.read_csv('data/selected_features_train.csv', index_col=0)
y_train = pd.read_csv('data/y_train.csv', index_col=0)
test_df = pd.read_csv('data/selected_features_test.csv', index_col=0)

def baseline_model():
    model = Sequential()
    model.add(Dense(100, activation="relu", input_shape=(336,)))
    model.add(BatchNormalization())
    model.add(Dense(4, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def training():

    model = baseline_model()

    model.fit(X_train, y_train, epochs=20, batch_size=128, verbose=1)

    model.save('models_test/nn_best_feature.h5')

    predictions = model.predict(X_train)
    preds = np.argmax(predictions, axis=1)
    pred_df = pd.DataFrame(preds, columns=['label'])
    print("f1_score", f1_score(pred_df, y_train, average = 'macro'))

training()

def predicition():

    model = load_model("models_test/nn_best_feature.h5")

    predictions = model.predict(test_df)

    preds = np.argmax(predictions, axis=1)

    pred_df = pd.DataFrame(preds, columns=['label'])

    pred_df.to_csv("predictions_test/nn_best_feature.csv")

predicition()
