import os
import pandas as pd

FOLDER_PATH = r"predictions_test"

def join_preds(input_file, folder_path=FOLDER_PATH):
    """
    Takes every prediction file in folder and votes best label depeding on predictions in folder
    """
    i = 0
    predictions = pd.DataFrame()
    for root,dirs,files in os.walk(folder_path):
        for file in files:
            df = pd.read_csv(folder_path + '\\' + file)
            predictions['file_' + str(i)] = df['label']
            i+=1
    predictions = predictions.mode(axis=1)

    predictions = predictions.rename(columns={0: "label"})

    pred_df = pd.DataFrame(predictions.iloc[:,0], columns=['label'])
    pred_df.label = pred_df.label.astype(int)

    pred_df.to_csv(input_file, index=True, index_label='Id')

join_preds("final_prediction_test.csv")

