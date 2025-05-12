import commons.mysql.mysqlHelper as sqlHelper
import mysql.connector
import pickle
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import Variable as var
import os

# This preparation dataset is for preparing the dataset as a memory object and able to be saved as physical file and
# it could be loaded into memory by pickle library when it is needed.
# This could save some time for retrieving the data from dataset.

# Dataset: std_label

sql_stmt = """
    Select
        CASE
            WHEN std_label = 'Neutral' THEN 0
            WHEN std_label = 'Happy' THEN 1
            WHEN std_label = 'Fear' THEN 2
            WHEN std_label = 'Surprise' THEN 3
            WHEN std_label = 'Angry' THEN 4
            WHEN std_label = 'Sad' THEN 5
            ELSE '-1'
        END AS std_label,  
        cleanedtxt,  
        translate_chn,  
        translate_my,  
        translate_tm,  
        cm_en_chn,  
        cm_en_my,  
        cm_en_tm,  
        cm_chn_en,  
        cm_chn_my,  
        cm_chn_tm,  
        cm_my_en,  
        cm_my_chn,  
        cm_my_tm,  
        cm_tm_en,  
        cm_tm_chn,  
        cm_tm_my,  
        cw_en_chn,  
        cw_en_my,  
        cw_en_tm,  
        cw_chn_en,  
        cw_chn_my,  
        cw_chn_tm,  
        cw_my_en,  
        cw_my_chn,  
        cw_my_tm,  
        cw_tm_en,  
        cw_tm_chn,  
        cw_tm_my,
        CASE
            WHEN pseudo_label = 'Neutral' THEN 0
            WHEN pseudo_label = 'Happy' THEN 1
            WHEN pseudo_label = 'Fear' THEN 2
            WHEN pseudo_label = 'Surprise' THEN 3
            WHEN pseudo_label = 'Angry' THEN 4
            WHEN pseudo_label = 'Sad' THEN 5
            ELSE '-1'
        END AS pseudo_label  
    from Synth_text
"""

class SrcData:
    std_label = -1
    pseudo_label = -1

    cleanedTxt = ""
    translate_chn = ""
    translate_my = ""
    translate_tm = ""

    cm_en_chn = ""
    cm_en_my = ""
    cm_en_tm = ""

    cm_chn_en = ""
    cm_chn_my = ""
    cm_chn_tm = ""

    cm_my_en = ""
    cm_my_chn = ""
    cm_my_tm = ""

    cm_tm_en = ""
    cm_tm_chn = ""
    cm_tm_my = ""

    cw_en_chn = ""
    cw_en_my = ""
    cw_en_tm = ""

    cw_chn_en = ""
    cw_chn_my = ""
    cw_chn_tm = ""

    cw_my_en = ""
    cw_my_chn = ""
    cw_my_tm = ""

    cw_tm_en = ""
    cw_tm_chn = ""
    cw_tm_my = ""

    def __init__(self,
                 std_label, cleanedTxt, translate_chn, translate_my, translate_tm,
                 cm_en_chn, cm_en_my, cm_en_tm,
                 cm_chn_en, cm_chn_my, cm_chn_tm,
                 cm_my_en, cm_my_chn, cm_my_tm,
                 cm_tm_en, cm_tm_chn, cm_tm_my,
                 cw_en_chn, cw_en_my, cw_en_tm,
                 cw_chn_en, cw_chn_my, cw_chn_tm,
                 cw_my_en, cw_my_chn, cw_my_tm,
                 cw_tm_en, cw_tm_chn, cw_tm_my, pseudo_label
                 ):
        self.std_label = std_label
        self.cleanedTxt = cleanedTxt
        self.translate_chn = translate_chn
        self.translate_my = translate_my
        self.translate_tm = translate_tm

        self.cm_en_chn = cm_en_chn
        self.cm_en_my = cm_en_my
        self.cm_en_tm = cm_en_tm

        self.cm_chn_en = cm_chn_en
        self.cm_chn_my = cm_chn_my
        self.cm_chn_tm = cm_chn_tm

        self.cm_my_en = cm_my_en
        self.cm_my_chn = cm_my_chn
        self.cm_my_tm = cm_my_tm

        self.cm_tm_en = cm_tm_en
        self.cm_tm_chn = cm_tm_chn
        self.cm_tm_my = cm_tm_my

        self.cw_en_chn = cw_en_chn
        self.cw_en_my = cw_en_my
        self.cw_en_tm = cw_en_tm

        self.cw_chn_en = cw_chn_en
        self.cw_chn_my = cw_chn_my
        self.cw_chn_tm = cw_chn_tm

        self.cw_my_en = cw_my_en
        self.cw_my_chn = cw_my_chn
        self.cw_my_tm = cw_my_tm

        self.cw_tm_en = cw_tm_en
        self.cw_tm_chn = cw_tm_chn
        self.cw_tm_my = cw_tm_my

        self.pseudo_label = pseudo_label


def get_data_db():
    """
    Get the data from database
    :return: data list
    """
    srcdatalist = []
    try:
        conn = sqlHelper.get_mysql_conn()
        mycursor = conn.cursor()

        mycursor.execute(sql_stmt)
        result = mycursor.fetchall()

        for i in result:
            std_label = i[0]

            cleanedTxt = i[1]
            translate_chn = i[2]
            translate_my = i[3]
            translate_tm = i[4]

            cm_en_chn = i[5]
            cm_en_my = i[6]
            cm_en_tm = i[7]

            cm_chn_en = i[8]
            cm_chn_my = i[9]
            cm_chn_tm = i[10]

            cm_my_en = i[11]
            cm_my_chn = i[12]
            cm_my_tm = i[13]

            cm_tm_en = i[14]
            cm_tm_chn = i[15]
            cm_tm_my = i[16]

            cw_en_chn = i[17]
            cw_en_my = i[18]
            cw_en_tm = i[19]

            cw_chn_en = i[20]
            cw_chn_my = i[21]
            cw_chn_tm = i[22]

            cw_my_en = i[23]
            cw_my_chn = i[24]
            cw_my_tm = i[25]

            cw_tm_en = i[26]
            cw_tm_chn = i[27]
            cw_tm_my = i[28]

            pseudo_label = i[29]

            data = SrcData(std_label, cleanedTxt, translate_chn, translate_my, translate_tm,
                           cm_en_chn, cm_en_my, cm_en_tm,
                           cm_chn_en, cm_chn_my, cm_chn_tm,
                           cm_my_en, cm_my_chn, cm_my_tm,
                           cm_tm_en, cm_tm_chn, cm_tm_my,
                           cw_en_chn, cw_en_my, cw_en_tm,
                           cw_chn_en, cw_chn_my, cw_chn_tm,
                           cw_my_en, cw_my_chn, cw_my_tm,
                           cw_tm_en, cw_tm_chn, cw_tm_my, pseudo_label)
            srcdatalist.append(data)

    except mysql.connector.Error as error:
        print("Failed to select record to database: {}".format(error))
    finally:
        if conn.is_connected():
            mycursor.close()
            conn.close()
            print("MySQL connection is closed")

    return srcdatalist


print("Start...")
# 1. Get the data from database
print("Step 1: Get the data from database...Start")
thelist = get_data_db()

print("Step 1: Get the data from database...End")

# 2. Generate English dataset
print("Step 2: Generate a dataset consists of English only...Start")
engdb_txt = []
engdb_label = []
engdb_plabel = []
for dt in thelist:
    engdb_txt.append(dt.cleanedTxt)
    engdb_label.append(dt.std_label)
    engdb_plabel.append(dt.pseudo_label)
print("Step 2: Generate a dataset consists of English only...End")

# 3. Generate Multilingual dataset
print("Step 3: Generate a dataset consists of Multilingual...Start")
multiLang_txt = []
multiLang_label = []
multiLang_plabel = []
for dt in thelist:
    # cleanedtxt,
    multiLang_txt.append(dt.cleanedTxt)
    multiLang_label.append(dt.std_label)
    multiLang_plabel.append(dt.pseudo_label)
    # translate_chn,
    multiLang_txt.append(dt.translate_chn)
    multiLang_label.append(dt.std_label)
    multiLang_plabel.append(dt.pseudo_label)
    # translate_my,
    multiLang_txt.append(dt.translate_my)
    multiLang_label.append(dt.std_label)
    multiLang_plabel.append(dt.pseudo_label)
    # translate_tm,
    multiLang_txt.append(dt.translate_tm)
    multiLang_label.append(dt.std_label)
    multiLang_plabel.append(dt.pseudo_label)
    # cm_en_chn,
    multiLang_txt.append(dt.cm_en_chn)
    multiLang_label.append(dt.std_label)
    multiLang_plabel.append(dt.pseudo_label)
    # cm_en_my,
    multiLang_txt.append(dt.cm_en_my)
    multiLang_label.append(dt.std_label)
    multiLang_plabel.append(dt.pseudo_label)
    # cm_en_tm,
    multiLang_txt.append(dt.cm_en_tm)
    multiLang_label.append(dt.std_label)
    multiLang_plabel.append(dt.pseudo_label)
    # cm_chn_en,
    multiLang_txt.append(dt.cm_chn_en)
    multiLang_label.append(dt.std_label)
    multiLang_plabel.append(dt.pseudo_label)
    # cm_chn_my,
    multiLang_txt.append(dt.cm_chn_my)
    multiLang_label.append(dt.std_label)
    multiLang_plabel.append(dt.pseudo_label)
    # cm_chn_tm,
    multiLang_txt.append(dt.cm_chn_tm)
    multiLang_label.append(dt.std_label)
    multiLang_plabel.append(dt.pseudo_label)
    # cm_my_en,
    multiLang_txt.append(dt.cm_my_en)
    multiLang_label.append(dt.std_label)
    multiLang_plabel.append(dt.pseudo_label)
    # cm_my_chn,
    multiLang_txt.append(dt.cm_my_chn)
    multiLang_label.append(dt.std_label)
    multiLang_plabel.append(dt.pseudo_label)
    # cm_my_tm,
    multiLang_txt.append(dt.cm_my_tm)
    multiLang_label.append(dt.std_label)
    multiLang_plabel.append(dt.pseudo_label)
    # cm_tm_en,
    multiLang_txt.append(dt.cm_tm_en)
    multiLang_label.append(dt.std_label)
    multiLang_plabel.append(dt.pseudo_label)
    # cm_tm_chn,
    multiLang_txt.append(dt.cm_tm_chn)
    multiLang_label.append(dt.std_label)
    multiLang_plabel.append(dt.pseudo_label)
    # cm_tm_my,
    multiLang_txt.append(dt.cm_tm_my)
    multiLang_label.append(dt.std_label)
    multiLang_plabel.append(dt.pseudo_label)
    # cw_en_chn,
    multiLang_txt.append(dt.cw_en_chn)
    multiLang_label.append(dt.std_label)
    multiLang_plabel.append(dt.pseudo_label)
    # cw_en_my,
    multiLang_txt.append(dt.cw_en_my)
    multiLang_label.append(dt.std_label)
    multiLang_plabel.append(dt.pseudo_label)
    # cw_en_tm,
    multiLang_txt.append(dt.cw_en_tm)
    multiLang_label.append(dt.std_label)
    multiLang_plabel.append(dt.pseudo_label)
    # cw_chn_en,
    multiLang_txt.append(dt.cw_chn_en)
    multiLang_label.append(dt.std_label)
    multiLang_plabel.append(dt.pseudo_label)
    # cw_chn_my,
    multiLang_txt.append(dt.cw_chn_my)
    multiLang_label.append(dt.std_label)
    multiLang_plabel.append(dt.pseudo_label)
    # cw_chn_tm,
    multiLang_txt.append(dt.cw_chn_tm)
    multiLang_label.append(dt.std_label)
    multiLang_plabel.append(dt.pseudo_label)
    # cw_my_en,
    multiLang_txt.append(dt.cw_my_en)
    multiLang_label.append(dt.std_label)
    multiLang_plabel.append(dt.pseudo_label)
    # cw_my_chn,
    multiLang_txt.append(dt.cw_my_chn)
    multiLang_label.append(dt.std_label)
    multiLang_plabel.append(dt.pseudo_label)
    # cw_my_tm,
    multiLang_txt.append(dt.cw_my_tm)
    multiLang_label.append(dt.std_label)
    multiLang_plabel.append(dt.pseudo_label)
    # cw_tm_en,
    multiLang_txt.append(dt.cw_tm_en)
    multiLang_label.append(dt.std_label)
    multiLang_plabel.append(dt.pseudo_label)
    # cw_tm_chn,
    multiLang_txt.append(dt.cw_tm_chn)
    multiLang_label.append(dt.std_label)
    multiLang_plabel.append(dt.pseudo_label)
    # cw_tm_my
    multiLang_txt.append(dt.cw_tm_my)
    multiLang_label.append(dt.std_label)
    multiLang_plabel.append(dt.pseudo_label)
print("Step 3: Generate a dataset consists of Multilingual...End")

# 4.
## This section is for creating a dataset which synthetic text are placed as individual columns
print("Step 4: Generate a dataset consists of combine...Start")
multiLangComb_txt = []
multiLangComb_label = []
multiLangComb_plabel = []
for dt in thelist:
    theRow = []
    # cleanedtxt
    theRow.append(dt.cleanedTxt)
    # translate_chn,
    theRow.append(dt.translate_chn)
    # translate_my,
    theRow.append(dt.translate_my)
    # translate_tm,
    theRow.append(dt.translate_tm)
    # cm_en_chn,
    theRow.append(dt.cm_en_chn)
    # cm_en_my,
    theRow.append(dt.cm_en_my)
    # cm_en_tm,
    theRow.append(dt.cm_en_tm)
    # cm_chn_en,
    theRow.append(dt.cm_chn_en)
    # cm_chn_my,
    theRow.append(dt.cm_chn_my)
    # cm_chn_tm,
    theRow.append(dt.cm_chn_tm)
    # cm_my_en,
    theRow.append(dt.cm_my_en)
    # cm_my_chn,
    theRow.append(dt.cm_my_chn)
    # cm_my_tm,
    theRow.append(dt.cm_my_tm)
    # cm_tm_en,
    theRow.append(dt.cm_tm_en)
    # cm_tm_chn,
    theRow.append(dt.cm_tm_chn)
    # cm_tm_my,
    theRow.append(dt.cm_tm_my)
    # cw_en_chn,
    theRow.append(dt.cw_en_chn)
    # cw_en_my,
    theRow.append(dt.cw_en_my)
    # cw_en_tm,
    theRow.append(dt.cw_en_tm)
    # cw_chn_en,
    theRow.append(dt.cw_chn_en)
    # cw_chn_my,
    theRow.append(dt.cw_chn_my)
    # cw_chn_tm,
    theRow.append(dt.cw_chn_tm)
    # cw_my_en,
    theRow.append(dt.cw_my_en)
    # cw_my_chn,
    theRow.append(dt.cw_my_chn)
    # cw_my_tm,
    theRow.append(dt.cw_my_tm)
    # cw_tm_en,
    theRow.append(dt.cw_tm_en)
    # cw_tm_chn,
    theRow.append(dt.cw_tm_chn)
    # cw_tm_my
    theRow.append(dt.cw_tm_my)

    multiLangComb_txt.append(np.array(theRow))
    multiLangComb_label.append(dt.std_label)
    multiLangComb_plabel.append(dt.pseudo_label)
print("Step 4: Generate a dataset consists of combine...End")


# 7. Save it into physical files
print("Step 5: Save the files...Start")

isExist = os.path.exists(var.DIR_OUTPUT)
if not isExist:
    os.makedirs(var.DIR_OUTPUT)
    print("Directory created successfully!")

# ORI - start
# ORI means there is no sampling involved.
#=======
fileObj = open(var.FILE_NOSAMPLING_ENG_TEXT_DATASET_LBL, 'wb')
pickle.dump(engdb_txt, fileObj)
fileObj.close()
fileObj = open(var.FILE_NOSAMPLING_ENG_LABEL_DATASET_LBL, 'wb')
pickle.dump(engdb_label, fileObj)
fileObj.close()

fileObj = open(var.FILE_NOSAMPLING_MULTI_TEXT_DATASET_LBL, 'wb')
pickle.dump(multiLang_txt, fileObj)
fileObj.close()
fileObj = open(var.FILE_NOSAMPLING_MULTI_LABEL_DATASET_LBL, 'wb')
pickle.dump(multiLang_label, fileObj)
fileObj.close()

fileObj = open(var.FILE_NOSAMPLING_MULTICOMB_TEXT_DATASET_LBL, 'wb')
pickle.dump(multiLangComb_txt, fileObj)
fileObj.close()
fileObj = open(var.FILE_NOSAMPLING_MULTICOMB_LABEL_DATASET_LBL, 'wb')
pickle.dump(multiLangComb_label, fileObj)
fileObj.close()

# ORI - end

print("Step 7: Save the files...End")

print("Done...")
