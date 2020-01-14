"""
First preprocess, uncomment last lines to execute preprocessing
"""
import pandas as pd
pd.options.display.max_seq_items = None
pd.set_option('display.max_columns', None)

from sklearn.preprocessing import MultiLabelBinarizer

import dateutil

import re

def normalize_data(train_df, test_df, columns):
    """Normalize data between 0 and 1"""
    train_df_to_normalize = train_df[columns]
    test_df_to_normalize = test_df[columns]

    train_df_min, train_df_max = train_df_to_normalize.min(), train_df_to_normalize.max()
    normalized_train_df = (train_df_to_normalize-train_df_min)/(train_df_max-train_df_min)
    normalized_test_df = (test_df_to_normalize-train_df_min)/(train_df_max-train_df_min)

    return normalized_train_df, normalized_test_df

def process_date(df):

    new_df = df.copy()
    new_df['Year'] = new_df.date.map(lambda date: dateutil.parser.parse(date, fuzzy=True).year)
    new_df['Month'] = new_df.date.map(lambda date: (dateutil.parser.parse(date, fuzzy=True).month-1)/11)
    new_df['Day'] = new_df.date.map(lambda date: (dateutil.parser.parse(date, fuzzy=True).day-1)/30)

    new_df['Hour'] = new_df.date.map(lambda date: dateutil.parser.parse(date, fuzzy=True).hour)
    new_df['Minute'] = new_df.date.map(lambda date: dateutil.parser.parse(date, fuzzy=True).minute)
    new_df['Second'] = new_df.date.map(lambda date: dateutil.parser.parse(date, fuzzy=True).second)

    new_df['Time'] = (new_df['Hour']*3600 + new_df['Minute']*60 + new_df['Second'])/86400

    new_df.drop(['date','Hour','Minute','Second'], axis= 1, inplace=True)

    return new_df

def one_hot_encode_mail_type(train_df,test_df,unique_word=True):
    train_x = train_df[['mail_type']]
    test_x = test_df[['mail_type']]

    union = pd.concat([train_x, test_x])

    if unique_word:
        union = union.mail_type.map(lambda mail_type: mail_type.lower().replace(" ", ""))

        one_hot = pd.get_dummies(union, prefix = 'mail_type_')

    else:
        union = union.mail_type.map(lambda mail_type: re.findall(r"[\w']+", mail_type.lower().replace(" ", "")))

        mlb = MultiLabelBinarizer()

        one_hot = pd.DataFrame(mlb.fit_transform(union),
                       columns=mlb.classes_,
                       index=union.index)

        one_hot = one_hot.add_prefix('mail_type_')

    train_df.drop(['mail_type'],axis= 1, inplace=True)
    test_df.drop(['mail_type'],axis= 1, inplace=True)

    train_df = train_df.join(one_hot.iloc[:train_x.shape[0], :])
    test_df = test_df.join(one_hot.iloc[train_x.shape[0]:, :])

    return train_df, test_df

def one_hot_encode_org(train_df,test_df):
    train_x = train_df[['org']]
    test_x = test_df[['org']]

    train_x = train_x.org.map(lambda org: org.lower().replace(" ", ""))
    test_x = test_x.org.map(lambda org: org.lower().replace(" ", ""))

    union = pd.concat([train_x, test_x])

    one_hot = pd.get_dummies(union, prefix = 'org')

    train_df.drop(['org'], axis= 1, inplace=True)
    test_df.drop(['org'], axis= 1, inplace=True)

    train_df = train_df.join(one_hot.iloc[:train_x.shape[0], :])
    test_df = test_df.join(one_hot.iloc[train_x.shape[0]:, :])

    return train_df, test_df


def one_hot_encode_tld(train_df,test_df,unique_word=True):
    train_x = train_df[['tld']]
    test_x = test_df[['tld']]

    union = pd.concat([train_x, test_x])

    if unique_word:
        union = union.tld.map(lambda tld: tld.lower().replace(" ", ""))

        one_hot = pd.get_dummies(union, prefix = 'tld')
    else:
        union = union.tld.map(lambda tld: re.findall(r"[\w']+", tld.lower().replace(" ", "")))

        mlb = MultiLabelBinarizer()

        one_hot = pd.DataFrame(mlb.fit_transform(union),
                           columns=mlb.classes_,
                           index=union.index)

        one_hot = one_hot.add_prefix('tld_')

    train_df.drop(['tld'], axis= 1, inplace=True)
    test_df.drop(['tld'], axis= 1, inplace=True)

    train_df = train_df.join(one_hot.iloc[:train_x.shape[0], :])
    test_df = test_df.join(one_hot.iloc[train_x.shape[0]:, :])

    return train_df, test_df

def chars_division(datasets):
    """Creates new features: salutations/chars_by_body, images/chars_by_body,
    urls/chars_by_body, chars_in_subject/chars_by_body"""
    divider_columns = ['salutations','images','urls','chars_in_subject']
    for dataset in datasets:
        for divider in divider_columns:
            dataset['divide_by_' + divider] = dataset.apply(lambda row: row[divider]/row.chars_in_body, axis = 1)


def preprocessing(train_df,test_df,unique_words=True):
    """Complete preprocessing:
        - create columns for each part of date: year, month, day, Time in seconds
        - create new division features
        - normalize between 0 and 1
        - one_hot_encode mail_type, org and tld based on separate words in strings if unique_words=False
        and complete strings otherwise"""

    # Handle missing values
    train_df.fillna('NotANumberr', inplace=True)
    test_df.fillna('NotANumberr', inplace=True)

    normalization_columns = ['ccs','images','urls','chars_in_subject','chars_in_body','Year']
    divider_columns = ['salutations','images','urls','chars_in_subject']
    for divider in divider_columns:
        normalization_columns.append('divide_by_' + divider)

    train_df, test_df = process_date(train_df), process_date(test_df)

    chars_division([train_df,test_df])

    normalized_train_df, normalized_test_df = normalize_data(train_df, test_df, normalization_columns)

    train_df.drop(normalization_columns, axis= 1, inplace=True)
    test_df.drop(normalization_columns, axis= 1, inplace=True)

    train_df, test_df = train_df.join(normalized_train_df), test_df.join(normalized_test_df)


    train_df, test_df = one_hot_encode_mail_type(train_df,test_df,unique_words)

    train_df, test_df = one_hot_encode_org(train_df,test_df)

    train_df, test_df = one_hot_encode_tld(train_df,test_df,unique_words)

    X_train = train_df.drop(['label'],axis= 1)
    Y_train = pd.DataFrame(train_df['label'])

    return X_train, Y_train, test_df


# TO UNCOMMENT
# Create CSVs
train_df = pd.read_csv('data/train.csv', index_col=0)
test_df = pd.read_csv('data/test.csv', index_col=0)

X_train, y_train, test_df = preprocessing(train_df, test_df, unique_words=False)

print(X_train.head(1))
print(y_train.head(1))

y_train.to_csv("data/y_train_2.csv")
X_train.to_csv("data/X_train_with_division_non_unique_words_2.csv")
test_df.to_csv("data/test_df_with_division_non_unique_words_2.csv")



