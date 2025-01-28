from transformers import BertTokenizer, BertForSequenceClassification
import PTM_lib as ptm
import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import random
import Variable as var
import commons.mysql.mysqlHelper as sqlHelper
import mysql.connector

"""
This script is for fine tuning the mBERT model with the best learning rate. 
"""

LBL_TYPE = "LBL"

def is_model_done(model_name):
    print("Start to query ...")
    isFound = False;
    try:
        conn = sqlHelper.get_mysql_conn()
        mycursor = conn.cursor()
        Select_sql = """
                    select count(1) from model_training where model_name = %s
                    """
        val = (model_name,)
        mycursor.execute(Select_sql, val)
        result = mycursor.fetchone()
        isFound = result[0] > 0

        print("Finish query ...")
    except mysql.connector.Error as error:
        print("Failed to select record to database: {}".format(error))
    finally:
        if conn.is_connected():
            mycursor.close()
            conn.close()
            print("MySQL connection is closed")

    return isFound

def training_process(start_epochs, end_epochs, DATA_SIZE, LANG_TYPE, LEARNING_RATE):

    # Configuration for training
    # start_epochs = 0 # start with 0
    # end_epochs = 19 #19 is 20th
    EXP_TYPE = 'RO3'
    # DATA_SIZE = var.SAMPLING_400 #_400, _800
    # LANG_TYPE = var.LANG_TYPE_ENG #var.LANG_TYPE_ENG or var.LANG_TYPE_ENG
    BATCH_SIZE = 8

    # Tuning parameter
    NUM_CLASSES = len(var.Label_Code_Desc)
    #LEARNING_RATE = 3e-5 #0.00003
    SAMPLING_TYPE = var.SAMPLING_TYPE_ORI
    model_type = "Final-mBERT" + "_" + LBL_TYPE
    model_name = "bert-base-multilingual-uncased"

    print("Testing info: " + EXP_TYPE)
    print("Language: " + LANG_TYPE)
    print("Data Sampling Type: " + SAMPLING_TYPE)
    print("Data size each class: " + str(DATA_SIZE))
    print("Model: " + model_name)

    def step1_load_df(lang, size):
        file_name = ptm.get_DF_LBL_file_path(lang, size)
        print("DataFrame file: " + file_name)
        fileObj = open(file_name, 'rb')
        df = pickle.load(fileObj)
        fileObj.close()
        return df

    # Step 1: Load the data
    df = step1_load_df(LANG_TYPE, DATA_SIZE)
    print("Dataset size: " + str(len(df)))

    # Step2: Adjustment on the dataset
    training_txt = df[ptm.get_DF_column_name(LANG_TYPE)].tolist()
    training_label = df[var.COLUMN_NAME_STD_LABEL].tolist()
    training_label = [int(i) for i in training_label]

    # 3. Load BERT tokenizer and model
    model_id = model_type + '_' + LANG_TYPE + '_' + str(DATA_SIZE) + '_' + str(LEARNING_RATE)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=NUM_CLASSES)

    Result = is_model_done(model_id)
    #Result = False
    if Result != True:
        ptm.Main_trainingModel_batch(model_id, tokenizer, model, training_txt, training_label, BATCH_SIZE,
                                     LEARNING_RATE, start_epochs, end_epochs)
        print("===============")
        print(model_id + " - done")
        print("===============")
    else:
        print("===============")
        print(model_id + " - Skiped")
        print("===============")

LEARNING_RATE_LIST = [1e-5] # This is the best learning rate
DATA_SIZE_LIST = [var.SAMPLING_400, var.SAMPLING_1600, var.SAMPLING_2800]
LANG_TYPE_LIST = [var.LANG_TYPE_ENG, var.LANG_TYPE_MULTI]

for size in DATA_SIZE_LIST:
    for lang in LANG_TYPE_LIST:
        for learning_rate in LEARNING_RATE_LIST:
            training_process(0, 14, size, lang, learning_rate)

# Single execution
#training_process(2, 4, var.SAMPLING_2800, var.LANG_TYPE_ENG, 1e-07)