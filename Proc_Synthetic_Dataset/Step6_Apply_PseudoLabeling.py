import mysql
import commons.mysql.mysqlHelper as sqlHelper
import commons.emotion.identifyEmotion as emo

class MyDataSet:
    id = ""
    eng_text = ""
    plabel = ""
def Read_dataSet():
    datalist = []
    try:
        conn = sqlHelper.get_mysql_conn()
        mycursor = conn.cursor()
        sql = ("select id, cleanedtxt from Synth_text where pseudo_label is null limit 10000")
        mycursor.execute(sql)
        result = mycursor.fetchall()

        for i in result:

            data = MyDataSet()
            data.id = i[0]
            data.eng_text = i[1]
            data.plabel = emo.identifyEmotion(str(i[1]))
            datalist.append(data)

    except mysql.connector.Error as error:
        print("Failed to select record to database: {}".format(error))
    finally:
        if conn.is_connected():
            mycursor.close()
            conn.close()
            print("MySQL connection is closed")

    return datalist


def update_dataset_pseudolabel(textDataList):

    try:
        conn = sqlHelper.get_mysql_conn()
        mycursor = conn.cursor()
        update_sql = "update Synth_text set pseudo_label = %s where id = %s "

        values = []

        for textData in textDataList:
            tuppleData = (textData.plabel, textData.id)
            values.append(tuppleData)

        # executemany() method
        mycursor.executemany(update_sql, values)
        # save changes
        conn.commit()

    except mysql.connector.Error as error:
        print("Failed to insert record to database: {}".format(error))
    finally:
        if conn.is_connected():
            mycursor.close()
            conn.close()
            print("MySQL connection is closed")


while True:
    """
    Always true to execute repeatly until all the data has been processed. This can be rerun if it stops half way.
    """
    dtList = Read_dataSet()
    if len(dtList) == 0:
        break

    update_dataset_pseudolabel(dtList)