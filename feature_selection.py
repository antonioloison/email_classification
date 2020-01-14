"""
Feature selection from lasso regularized logistic regression that selects important features
"""
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.max_seq_items = None

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

X_train = pd.read_csv('data/X_train_with_division_non_unique_words.csv', index_col=0)
y_train = pd.read_csv('data/y_train.csv', index_col=0)
test_df = pd.read_csv('data/test_df_with_division_non_unique_words.csv', index_col=0)

# Set the regularization parameter C=1
logistic = LogisticRegression(C=1, penalty="l1", random_state=7).fit(X_train, y_train)
model = SelectFromModel(logistic, prefit=True)
print("Model trained")

X_new = model.transform(X_train)
test_new = model.transform(test_df)

# Get back the kept features as a DataFrame with dropped columns as all 0s
selected_features_train = pd.DataFrame(model.inverse_transform(X_new),
                                 index=X_train.index,
                                 columns=X_train.columns)
selected_features_test = pd.DataFrame(model.inverse_transform(test_new),
                                 index=test_df.index,
                                 columns=X_train.columns)
print("Features selected")

# Dropped columns have values of all 0s, keep other columns
selected_columns = selected_features_train.columns[selected_features_train.var() != 0]

selected_features_train = selected_features_train[selected_columns]
selected_features_test = selected_features_test[selected_columns]

selected_features_train.to_csv('data/selected_features_train.csv')
selected_features_test.to_csv('data/selected_features_test.csv')
