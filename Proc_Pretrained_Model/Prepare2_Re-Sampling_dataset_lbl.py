import Variable as var
import pickle
import pandas as pd
import random
import os
from imblearn.over_sampling import RandomOverSampler

# 1. Retrieve the dataset multi combine

fileName = var.FILE_NOSAMPLING_MULTICOMB_TEXT_DATASET_LBL
print(fileName)

isExist = os.path.exists(fileName)
print(isExist)

fileObj = open(var.FILE_NOSAMPLING_MULTICOMB_TEXT_DATASET_LBL, 'rb')
multi_combined_db_nos_txt_1 = pickle.load(fileObj)
fileObj.close()
fileObj = open(var.FILE_NOSAMPLING_MULTICOMB_LABEL_DATASET_LBL, 'rb')
multi_combined_db_nos_label_1 = pickle.load(fileObj)
fileObj.close()


# 2. Build a dataframe
txt_file = ''
label_file = ''

cleanedtxt_list = []
translate_chn_list = []
translate_my_list = []
translate_tm_list = []
cm_en_chn_list = []
cm_en_my_list = []
cm_en_tm_list = []
cm_chn_en_list = []
cm_chn_my_list = []
cm_chn_tm_list = []
cm_my_en_list = []
cm_my_chn_list = []
cm_my_tm_list = []
cm_tm_en_list = []
cm_tm_chn_list = []
cm_tm_my_list = []
cw_en_chn_list = []
cw_en_my_list = []
cw_en_tm_list = []
cw_chn_en_list = []
cw_chn_my_list = []
cw_chn_tm_list = []
cw_my_en_list = []
cw_my_chn_list = []
cw_my_tm_list = []
cw_tm_en_list = []
cw_tm_chn_list = []
cw_tm_my_list = []

for item in multi_combined_db_nos_txt_1:
    cleanedtxt_list.append(item[0])
    translate_chn_list.append(item[1])
    translate_my_list.append(item[2])
    translate_tm_list.append(item[3])
    cm_en_chn_list.append(item[4])

    cm_en_my_list.append(item[5])
    cm_en_tm_list.append(item[6])
    cm_chn_en_list.append(item[7])
    cm_chn_my_list.append(item[8])
    cm_chn_tm_list.append(item[9])

    cm_my_en_list.append(item[10])
    cm_my_chn_list.append(item[11])
    cm_my_tm_list.append(item[12])
    cm_tm_en_list.append(item[13])
    cm_tm_chn_list.append(item[14])

    cm_tm_my_list.append(item[15])
    cw_en_chn_list.append(item[16])
    cw_en_my_list.append(item[17])
    cw_en_tm_list.append(item[18])
    cw_chn_en_list.append(item[19])

    cw_chn_my_list.append(item[20])
    cw_chn_tm_list.append(item[21])
    cw_my_en_list.append(item[22])
    cw_my_chn_list.append(item[23])
    cw_my_tm_list.append(item[24])

    cw_tm_en_list.append(item[25])
    cw_tm_chn_list.append(item[26])
    cw_tm_my_list.append(item[27])

dict = {
    'std_label': multi_combined_db_nos_label_1,
    'cleanedtxt':cleanedtxt_list,
    'translate_chn':translate_chn_list,
    'translate_my':translate_my_list,
    'translate_tm':translate_tm_list,
    'cm_en_chn':cm_en_chn_list,
    'cm_en_my':cm_en_my_list,
    'cm_en_tm':cm_en_tm_list,
    'cm_chn_en':cm_chn_en_list,
    'cm_chn_my':cm_chn_my_list,
    'cm_chn_tm':cm_chn_tm_list,
    'cm_my_en':cm_my_en_list,
    'cm_my_chn':cm_my_chn_list,
    'cm_my_tm':cm_my_tm_list,
    'cm_tm_en':cm_tm_en_list,
    'cm_tm_chn':cm_tm_chn_list,
    'cm_tm_my':cm_tm_my_list,
    'cw_en_chn':cw_en_chn_list,
    'cw_en_my':cw_en_my_list,
    'cw_en_tm':cw_en_tm_list,
    'cw_chn_en':cw_chn_en_list,
    'cw_chn_my':cw_chn_my_list,
    'cw_chn_tm':cw_chn_tm_list,
    'cw_my_en':cw_my_en_list,
    'cw_my_chn':cw_my_chn_list,
    'cw_my_tm':cw_my_tm_list,
    'cw_tm_en':cw_tm_en_list,
    'cw_tm_chn':cw_tm_chn_list,
    'cw_tm_my':cw_tm_my_list
}

df = pd.DataFrame(dict)


# 3. Re sampling process - English
sampling_class_size = [var.SAMPLING_400, var.SAMPLING_800, var.SAMPLING_1200, var.SAMPLING_1600, var.SAMPLING_2000, var.SAMPLING_2400, var.SAMPLING_2800]
#sampling_class_size = [var.SAMPLING_3200, var.SAMPLING_3600]
emotion_list = list(var.Label_Code_Desc.keys())

# 20240901 - for pseudo labeling
label_type_list = ['']

for sampling_size in sampling_class_size:
#for sampling_size in [10]:

    # create empty dataframe with the column names
    columnsList = ['std_label']
    columnsList = columnsList.append(var.language_type)
    sampling_df = pd.DataFrame(columns=columnsList)
    print("Working on: " + str(sampling_size))

    # Working on each column
    for emotion in emotion_list:
        print("  Working on: " + var.Label_Code_Desc.get(emotion))
        filtered_df = df[df['std_label'] == str(emotion)]

        if len(filtered_df) > sampling_size:
            subSample_size = sampling_size
        else:
            subSample_size = len(filtered_df)

        selected_df = filtered_df.sample(n=subSample_size)
        sampling_df = pd.concat([sampling_df, selected_df], ignore_index=True)

    print(sampling_df)

    # Start to do sampling
    X_train = sampling_df.drop('std_label', axis=1)
    y_train = sampling_df['std_label'].tolist()
    oversampler = RandomOverSampler(random_state=42)

    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

    resampled_df = pd.DataFrame(X_train_resampled, columns=X_train.columns)
    resampled_df['std_label'] = y_train_resampled

    print(resampled_df.head(5))

    file_name = var.FILE_MULTICOMB_DF_LBL + str(sampling_size) + ".obj"
    fileObj = open(file_name, 'wb')
    pickle.dump(sampling_df, fileObj)
    fileObj.close()

