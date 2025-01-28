import pickle
import pandas as pd
import Variable as var


def unique(list1):
    """
    Get the unique items from the list
    :param list1: Provided list
    :return: Unique items list
    """
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    unique_list = [int(i) for i in unique_list]
    for x in unique_list:
        print(type(x))
        print(str(x) + ' ' + var.Label_Code_Desc[x])


print("Loading dataset ... start")
## Over sampling
# Example of loading data for English dataset
fileObj = open(var.FILE_OVERSAMPLING_ENG_TEXT_DATASET_LBL, 'rb')
engdb_over_txt_1 = pickle.load(fileObj)
fileObj.close()
fileObj = open(var.FILE_OVERSAMPLING_ENG_LABEL_DATASET_LBL, 'rb')
engdb_over_label_1 = pickle.load(fileObj)
fileObj.close()

# Example of loading data for Multilingual dataset
fileObj = open(var.FILE_OVERSAMPLING_MULTI_TEXT_DATASET_LBL, 'rb')
multidb_over_txt_1 = pickle.load(fileObj)
fileObj.close()
fileObj = open(var.FILE_OVERSAMPLING_MULTI_LABEL_DATASET_LBL, 'rb')
multidb_over_label_1 = pickle.load(fileObj)
fileObj.close()

# Example of loading data for Multilingual combined dataset
fileObj = open(var.FILE_OVERSAMPLING_MULTICOMB_TEXT_DATASET_LBL, 'rb')
multi_combined_db_over_txt_1 = pickle.load(fileObj)
fileObj.close()
fileObj = open(var.FILE_OVERSAMPLING_MULTICOMB_LABEL_DATASET_LBL, 'rb')
multi_combined_db_over_label_1 = pickle.load(fileObj)
fileObj.close()

## Under sampling
# Example of loading data for English dataset
fileObj = open(var.FILE_UNDERSAMPLING_ENG_TEXT_DATASET_LBL, 'rb')
engdb_under_txt_1 = pickle.load(fileObj)
fileObj.close()
fileObj = open(var.FILE_UNDERSAMPLING_ENG_LABEL_DATASET_LBL, 'rb')
engdb_under_label_1 = pickle.load(fileObj)
fileObj.close()

# Example of loading data for Multilingual dataset
fileObj = open(var.FILE_UNDERSAMPLING_MULTI_TEXT_DATASET_LBL, 'rb')
multidb_under_txt_1 = pickle.load(fileObj)
fileObj.close()
fileObj = open(var.FILE_UNDERSAMPLING_MULTI_LABEL_DATASET_LBL, 'rb')
multidb_under_label_1 = pickle.load(fileObj)
fileObj.close()

# Example of loading data for Multilingual combined dataset
fileObj = open(var.FILE_UNDERSAMPLING_MULTICOMB_TEXT_DATASET_LBL, 'rb')
multi_combined_db_under_txt_1 = pickle.load(fileObj)
fileObj.close()
fileObj = open(var.FILE_UNDERSAMPLING_MULTICOMB_LABEL_DATASET_LBL, 'rb')
multi_combined_db_under_label_1 = pickle.load(fileObj)
fileObj.close()

print("Loading dataset ... end")

# 6. Info
print("The unique labels in English (over sampling): ")
unique(engdb_over_label_1)
print("The unique labels in Multilingual (over sampling): ")
unique(multidb_over_label_1)
print("The unique labels in Multilingual Combine (over sampling): ")
unique(multi_combined_db_over_label_1)

print("The unique labels in English (under sampling): ")
unique(engdb_under_label_1)
print("The unique labels in Multilingual (under sampling): ")
unique(multidb_under_label_1)
print("The unique labels in Multilingual Combine (under sampling): ")
unique(multi_combined_db_under_label_1)


count = pd.Series(engdb_over_label_1).value_counts()
print("For Eng (over sampling):")
print(count)
count = pd.Series(multidb_over_label_1).value_counts()
print("For Multilingual (over sampling):")
print(count)
count = pd.Series(multi_combined_db_over_label_1).value_counts()
print("For Multilingual Combine (over sampling):")
print(count)

count = pd.Series(engdb_under_label_1).value_counts()
print("For Eng (under sampling):")
print(count)
count = pd.Series(multidb_under_label_1).value_counts()
print("For Multilingual (under sampling):")
print(count)
count = pd.Series(multi_combined_db_under_label_1).value_counts()
print("For Multilingual Combine (under sampling):")
print(count)

