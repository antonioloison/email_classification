import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.max_seq_items = None

from utils import purge_text, parse_date, reformat_date

from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, BatchNormalization, ReLU, PReLU, Dropout, Concatenate, Softmax, Reshape
from keras.optimizers import Adam
from keras.regularizers import l2

from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras.wrappers.scikit_learn import KerasClassifier

from voting_classifier.vote_from_csvs import join_preds



VOCAB_SIZE = 336-15
EMBEDDED = 8
MAX_LENGTH = 6
COLUMNS = ['mail_type_alternative', 'mail_type_html', 'mail_type_mixed',
       'mail_type_notanumberr', 'mail_type_plain', 'mail_type_related',
       'mail_type_text', 'org_academia-mail', 'org_account', 'org_agents',
       'org_airpostmail', 'org_airtable', 'org_airtable-2', 'org_airvistara',
       'org_amazon', 'org_angel', 'org_applemusic', 'org_arya', 'org_asana',
       'org_aspiringminds', 'org_asuswebstorage', 'org_autodesk',
       'org_awseducate', 'org_bigbasket', 'org_booking', 'org_bookmyforex',
       'org_box8letters', 'org_brandshop', 'org_brilliant',
       'org_caisse-epargne', 'org_cambridge-intelligence', 'org_careers360',
       'org_cbsnewsletter', 'org_centralesupelec', 'org_change',
       'org_citruspay', 'org_classroom', 'org_cleartrip', 'org_cloudhq',
       'org_cloudns', 'org_clubvistara', 'org_cocubes', 'org_codalab',
       'org_codeanywhere', 'org_codecademy', 'org_codechef', 'org_codeschool',
       'org_connect', 'org_coupondunia', 'org_coursera', 'org_crazydomains',
       'org_crowdai', 'org_dailyspeak', 'org_datacamp',
       'org_dataquest-0740d8e6e13d', 'org_deeplearning',
       'org_dexterousdigitech', 'org_digital', 'org_digitalglobe-platform',
       'org_discuss', 'org_diux', 'org_docker', 'org_dominos', 'org_donate',
       'org_doodle', 'org_dropbox', 'org_dropboxmail', 'org_duolingo', 'org_e',
       'org_ebay', 'org_ecp', 'org_edm', 'org_edx', 'org_eer', 'org_em',
       'org_email', 'org_emailer', 'org_engage', 'org_entropay', 'org_et',
       'org_evernote', 'org_facebookmail', 'org_flat', 'org_flipkart',
       'org_flipkartletters', 'org_flipkartpromotions', 'org_foursquare',
       'org_freecharge', 'org_freelancer', 'org_freshersworld',
       'org_freshworks', 'org_gaanainfo', 'org_getpostman', 'org_github',
       'org_glassdoor', 'org_gmail', 'org_go', 'org_godaddy', 'org_goindigo',
       'org_google', 'org_gotinder', 'org_granular', 'org_hackerearth',
       'org_hackerrank', 'org_hackerrankmail', 'org_horosproject',
       'org_hotmail', 'org_htc', 'org_ieee', 'org_iheartdogs-email',
       'org_iiitd', 'org_imdb', 'org_imindmap', 'org_impactathon',
       'org_incontrolproductions', 'org_indiatimes', 'org_info',
       'org_innerchef', 'org_inoxmovies', 'org_inria', 'org_insideapple',
       'org_insure', 'org_internations', 'org_interviewbit', 'org_intl',
       'org_irctc', 'org_kaggle', 'org_khanacademy', 'org_kotak', 'org_limnu',
       'org_linkedin', 'org_localcirclesmail', 'org_lss', 'org_m2i',
       'org_magento', 'org_magoosh', 'org_mail', 'org_mak', 'org_makemytrip',
       'org_mapbox', 'org_marketing', 'org_mathworks', 'org_media',
       'org_medium', 'org_mentor', 'org_messages', 'org_microsoft',
       'org_monsterindia', 'org_msr-cmt', 'org_mysliderule', 'org_myworkday',
       'org_namomail', 'org_naylormarketing', 'org_neo4j', 'org_neotechnology',
       'org_neptune', 'org_new', 'org_news', 'org_newsgram', 'org_newsletter',
       'org_ni', 'org_notanumberr', 'org_notifications', 'org_nrsc',
       'org_nvidia', 'org_olacabs', 'org_oneplus', 'org_oneplusstore',
       'org_onlinerti', 'org_overleaf', 'org_paytm', 'org_paytmemail',
       'org_payu', 'org_phonepe', 'org_piazza', 'org_piazzacareers',
       'org_pinterest', 'org_pivotaltracker', 'org_planet', 'org_plus',
       'org_primevideo', 'org_pvrcinemas', 'org_quora', 'org_rapidapi',
       'org_readcube', 'org_realtimeboard', 'org_redwolf', 'org_reply',
       'org_repositoryhosting', 'org_research-mail', 'org_researchgate',
       'org_researchgatemail', 'org_ride', 'org_rit', 'org_royalsundaram',
       'org_rs-components', 'org_sampark', 'org_sbi', 'org_sg',
       'org_shining3d', 'org_shriramgeneral', 'org_shriramgi',
       'org_signalprocessingsociety', 'org_slack', 'org_smtpmailbox',
       'org_splitwise', 'org_spotify', 'org_springboard', 'org_stackoverflow',
       'org_statebankrewardz', 'org_studapart', 'org_student-cs',
       'org_supelec', 'org_swiggy', 'org_symless', 'org_teamviewer',
       'org_technolutions', 'org_thefork', 'org_thomascook', 'org_tikona',
       'org_tnt', 'org_topcoder', 'org_trello', 'org_trm', 'org_truecaller',
       'org_tufinawatches', 'org_twitter', 'org_udacity', 'org_ugcmailing',
       'org_unisys', 'org_updates', 'org_usebackpack', 'org_usief',
       'org_vincerowatches', 'org_virtuosos', 'org_vito', 'org_vodafone',
       'org_web-spicejet', 'org_witai-2', 'org_wooe', 'org_xoom', 'org_xprize',
       'org_youtube', 'org_zepass', 'org_zoom', 'org_zoomgroup', 'tld_ac',
       'tld_amazon', 'tld_apple', 'tld_aramex', 'tld_asus', 'tld_be',
       'tld_bird', 'tld_bitly', 'tld_bnpparibas', 'tld_bookmyshow',
       'tld_caisse', 'tld_cardekho', 'tld_change', 'tld_classmates', 'tld_co',
       'tld_com', 'tld_coursera', 'tld_crazydomains', 'tld_digitalglobe',
       'tld_dropbox', 'tld_ebay', 'tld_edu', 'tld_epargne', 'tld_evernote',
       'tld_findstay', 'tld_fr', 'tld_freelancer', 'tld_glassdoor',
       'tld_goodreads', 'tld_google', 'tld_gopro', 'tld_gov', 'tld_grammarly',
       'tld_hp', 'tld_ibm', 'tld_in', 'tld_info', 'tld_instagram',
       'tld_intercom', 'tld_io', 'tld_itunes', 'tld_jabong', 'tld_kotak',
       'tld_linkedin', 'tld_makemytrip', 'tld_mathworks', 'tld_microsoft',
       'tld_mil', 'tld_miscota', 'tld_mktg', 'tld_ml', 'tld_mozilla',
       'tld_net', 'tld_netflix', 'tld_notanumberr', 'tld_nvidia',
       'tld_onedrive', 'tld_org', 'tld_payback', 'tld_paypal', 'tld_paytm',
       'tld_pinterest', 'tld_pro', 'tld_pytorch', 'tld_reliancegeneral',
       'tld_sdconnect', 'tld_sky', 'tld_speakingtree', 'tld_supelec',
       'tld_tata', 'tld_tripadvisor', 'tld_vincerowatches', 'tld_wfp',
       'tld_windows', 'tld_xoom']

X_train = pd.read_csv('data/selected_features_train.csv', index_col=0)
y_train = pd.read_csv('data/y_train.csv', index_col=0)
test_df = pd.read_csv('data/selected_features_test.csv', index_col=0)

# print(X_train.iloc[:,15:].columns)
# print(X_train.shape)

def get_index(list1,list2):
    l = []
    for element in list1:
        l.append(list2.index(element)+1)
    return l

def labelize_row(row):
    column_names = row == 1
    serie = row[column_names]
    list_labels = list(serie.index)
    list_labels = get_index(list_labels, COLUMNS)
    n = len(list_labels)
    return list_labels + ((MAX_LENGTH - n)*[0])

# print(labelize_row(X_train.iloc[2,15:]))

def prepare_data_for_embedding(df):
    value_part = df.iloc[:,:15]
    value_part = value_part.values
    one_hot_part = df.iloc[:,15:]
    one_hot_part = one_hot_part.apply(lambda row: labelize_row(row), axis = 1)
    one_hot_part = one_hot_part.values
    one_hot_part_list = list(one_hot_part)
    one_hot_part_array = np.array([element for list in one_hot_part_list for element in list])
    n = one_hot_part_array.shape[0]//MAX_LENGTH
    one_hot_part_array = one_hot_part_array.reshape((n,MAX_LENGTH))
    return [one_hot_part_array, value_part]

# print(prepare_data_for_embedding(X_train.iloc[:30,:])[0].shape)

L = len(COLUMNS)
train_x = prepare_data_for_embedding(X_train.iloc[:,:])
print(train_x[0].shape)
print(train_x[1].shape)
train_y = y_train.values
test_x = prepare_data_for_embedding(test_df)

print("Data loaded")

scale = StandardScaler()
train_x[1] = scale.fit_transform(train_x[1])
test_x[1] = scale.transform(test_x[1])

DROPOUT = 0.3

def best_model():
    network_org_input = Input(shape=(MAX_LENGTH, ))
    network_org = Embedding(L + 1, 30)(network_org_input)
    network_org = Flatten()(network_org)

    network_values_input = Input(shape=(15, ))
    network_value = network_values_input

    network_merge = Concatenate()([network_org, network_value])
    network_merge = BatchNormalization()(network_merge)
    network_merge = Dense(25)(network_merge)
    network_merge = ReLU()(network_merge)
    network_merge = BatchNormalization()(network_merge)
    network_merge = Dense(4)(network_merge)
    network_merge = Softmax()(network_merge)

    model = Model(inputs=[network_org_input, network_values_input], outputs=network_merge)
    model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    return model


model = best_model()
model.fit(train_x, train_y, epochs=30, batch_size=128)
model.save('super_embedding.h5')


