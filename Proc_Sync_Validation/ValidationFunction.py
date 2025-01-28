import commons.lang.CleanText as ct
import commons.mysql.mysqlHelper as sqlHelper
from ValidationObject import SyncDataSet
import mysql
from sentence_transformers import SentenceTransformer, util


def get_Syn_Dataset(targetTable, recordline):
    oritxt_list = []
    Main_Query = """
        SELECT
            id, oritxt, cleanedtxt, 
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
            cw_tm_my
        from mydataset 
    """

    Sub_Query = " where length(cleanedtxt) > 0 and id not in (select id from " + targetTable + ") " + recordline + ";"
    Query_Str = Main_Query + Sub_Query

    # Get the oritxt from mysql

    try:
        conn = sqlHelper.get_mysql_conn()
        #print("MySQL connection is opened")
        mycursor = conn.cursor()
        mycursor.execute(Query_Str)
        myresult = mycursor.fetchall()

        i = 0

        for x in myresult:
            data = SyncDataSet(x[0], x[1], x[2], x[3], x[4],
                               x[5], x[6], x[7], x[8], x[9],
                               x[10], x[11], x[12], x[13], x[14],
                               x[15], x[16], x[17], x[18], x[19],
                               x[20], x[21], x[22], x[23], x[24],
                               x[25], x[26], x[27], x[28], x[29]
                               )
            oritxt_list.append(data)

    except mysql.connector.Error as error:
        print("Failed to select record to database: {}".format(error))
    finally:
        if conn.is_connected():
            mycursor.close()
            conn.close()
            #print("MySQL connection is closed")
    return oritxt_list

def getSimilarityBySentenceTransformer(model, text_1, text_2):
    embedding_1 = model.encode(text_1, convert_to_tensor=True)
    embedding_2 = model.encode(text_2, convert_to_tensor=True)

    # Compute cosine similarity
    similarity = util.cos_sim(embedding_1, embedding_2).item()

    # Output
    #print(f"Cosine Similarity: {similarity:.4f}")
    return round(similarity, 4)

def getSimilarityByBertScore(model, text_1, text_2):
    # Compute BERTScore
    P, R, F1 = model(cands=[text_2], refs=[text_1], lang="en", verbose=True)
    return P, R, F1

def insertUpdateSimilarityType1(tableName, id, fieldName, value):
    sql_msg = "INSERT INTO " + tableName + " (id, " + fieldName + ") VALUES(" + str(id) + ", " + str(value) + ") ON DUPLICATE KEY UPDATE "+fieldName+"=" + str(value)
    try:
        conn = sqlHelper.get_mysql_conn()
        #print("MySQL connection is opened")
        mycursor = conn.cursor()
        mycursor.execute(sql_msg)
        conn.commit()
    except mysql.connector.Error as error:
        print("Failed to insert record to database: {}".format(error))
    finally:
        if conn.is_connected():
            mycursor.close()
            conn.close()
            #print("MySQL connection is closed")


def insertUpdateSimilarityType2(tableName, id, fieldName, P, R, F1):
    sql_msg = ("INSERT INTO " + tableName + " (id, " + fieldName + "_prec, " + fieldName + "_rec, " + fieldName + "_f1) "+
               "VALUES(" + str(id) + ", " + str(P) + ", " + str(R) + ", " + str(F1) +
               ") ON DUPLICATE KEY UPDATE " + fieldName + "_prec=" + str(P) + ", " + fieldName + "_rec=" + str(R)+ ", " + fieldName + "_rec=" + str(R))
    try:
        conn = sqlHelper.get_mysql_conn()
        #print("MySQL connection is opened")
        mycursor = conn.cursor()
        mycursor.execute(sql_msg)

    except mysql.connector.Error as error:
        print("Failed to insert record to database: {}".format(error))
    finally:
        if conn.is_connected():
            mycursor.close()
            conn.close()
            #print("MySQL connection is closed")