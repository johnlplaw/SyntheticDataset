import Variable as var
import pickle
import pandas as pd

def create_training_eng (sampling_size):
    """
    Get the English text dataframe from the multicomb ori resampled dataset
    :param sampling_size: the sampling size
    :return: a dataframe with a samling size
    """
    # 1. Read the dataframe object
    source_file = var.FILE_MULTICOMB_DF_LBL + str(sampling_size) + ".obj"
    fileObj = open(source_file, 'rb')
    source_df = pickle.load(fileObj)
    fileObj.close()

    # 2. create the english dataframe
    new_df = source_df[[var.COLUMN_NAME_STD_LABEL,var.COLUMN_NAME_ENG_TXT]]

    return new_df

def create_training_multi (sampling_size):
    # 1. Read the dataframe object
    source_file = var.FILE_MULTICOMB_DF_LBL + str(sampling_size) + ".obj"
    fileObj = open(source_file, 'rb')
    source_df = pickle.load(fileObj)
    fileObj.close()

    # 2. Create multilingual dataframe
    new_df = pd.DataFrame(columns=[var.COLUMN_NAME_STD_LABEL, var.COLUMN_NAME_MULTI_LANG_TXT, var.COLUMN_NAME_SRC_TYPE, var.COLUMN_NAME_ENG_TXT])
    col_idx = 0
    source_df = source_df.reset_index()
    for index, row in source_df.iterrows():
        col_idx = col_idx + 1
        col_idx = col_idx % len(var.language_type)
        new_df.loc[len(new_df.index)] = [row[var.COLUMN_NAME_STD_LABEL], row[var.language_type[col_idx]], var.language_type[col_idx], row[var.COLUMN_NAME_ENG_TXT]]
    return new_df


for size in var.sampling_size_list:
#for size in [var.SAMPLING_3200, var.SAMPLING_3600]:
    print("Work on " + str(size))
    eng_df = create_training_eng(size)
    multi_df = create_training_multi(size)

    fileObj = open(var.FILE_TRAINING_ENG_NOS_LBL + str(size) + ".obj", 'wb')
    pickle.dump(eng_df, fileObj)
    fileObj.close()
    fileObj = open(var.FILE_TRAINING_MUL_NOS_LBL + str(size) + ".obj", 'wb')
    pickle.dump(multi_df, fileObj)
    fileObj.close()

# Sample to read the created dataset with a particular size
# size = 400
# fileObj = open(var.FILE_TRAINING_MUL_NOS + str(size) + ".obj", 'rb')
# multi_df = pickle.load(fileObj)
# fileObj.close()
