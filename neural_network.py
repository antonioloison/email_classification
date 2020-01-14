import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras import regularizers
pd.options.display.max_seq_items = None

lambda1 = 0.0003
lambda2 = 0.05
# Read csvs

# X_train = pd.read_csv('../data/selected_features_train.csv', index_col=0)
# y_train = pd.read_csv('../data/y_train.csv', index_col=0)
# test_df = pd.read_csv('../data/selected_features_test.csv', index_col=0)
# print(X_train.shape)
# #12 from mail, 21 from org
# X_train = X_train.iloc[:,:]
# # y_train = y_train.iloc[:,12:]
# test_df = test_df.iloc[:,:]
#
def baseline_model():
    model = Sequential()
    model.add(Dense(100, activation="relu", input_shape=(336,)))
    model.add(BatchNormalization())
    # model.add(Dropout(0.1))
    # model.add(Dense(25, activation="relu", input_shape=(X_train.shape[1],)))
    # model.add(BatchNormalization())
    model.add(Dense(4, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
#

# for i in range(20):
#     model = baseline_model()
#
#     model.fit(X_train, y_train, epochs=20, batch_size=128, verbose=1)
#
#     model.save('../best_models/nn_best_feature_' + str(i) + '.h5')

#
# predictions = model.predict(X_train)
# preds = np.argmax(predictions, axis=1)
# pred_df = pd.DataFrame(preds, columns=['label'])
# print("f1_score", f1_score(pred_df, y_train, average = 'macro'))
# predictions = model.predict(X_test)
# preds = np.argmax(predictions, axis=1)
# pred_df = pd.DataFrame(preds, columns=['label'])
# print("f1_score", f1_score(pred_df, y_test, average = 'macro'))
# model.save('dropout_no_reg_best_feature.h5')
# predictions = model.predict(test_df)
# preds = np.argmax(predictions, axis=1)
# pred_df = pd.DataFrame(preds, columns=['label'])
# pred_df.to_csv("dropout_no_reg_best_feature.csv", index=True, index_label='Id')

# print(Y_train)

# estimator = KerasClassifier(build_fn=baseline_model, epochs=25, batch_size=200, verbose=1)
# kfold = KFold(n_splits=10, shuffle=True)
# scores = cross_val_score(estimator, X_train, Y_train, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (scores.mean()*100, scores.std()*100))
