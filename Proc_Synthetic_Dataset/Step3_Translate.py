import time
from datetime import datetime

from tqdm import tqdm
from translate_shell.translate import translate

import commons.mysql.mysqlHelper as sqlHelper


# https://stackoverflow.com/questions/68275857/urllib-error-urlerror-urlopen-error-ssl-certificate-verify-failed-certifica


class SrcData:
    id = ""
    cleanedTxt = ""
    translate_chn = ""
    translate_my = ""
    translate_tm = ""

    def __init__(self, id, cleanedTxt, translate_chn, translate_my, translate_tm):
        self.id = id
        self.cleanedTxt = cleanedTxt
        self.translate_chn = translate_chn
        self.translate_my = translate_my
        self.translate_tm = translate_tm

    def __str__(self):
        return str(self.id) + " | " + self.cleanedTxt + " | " + self.translate_chn + " | " + str(
            self.translate_my) + " | " + str(self.translate_tm)


def execute():
    start_time = datetime.now()
    # -----------------

    # Get the oritxt from mysql
    conn = sqlHelper.get_mysql_conn()

    mycursor = conn.cursor()
    sql = "SELECT ID, cleanedTxt from Synth_text where translate_chn is null and not cleanedTxt = '' limit 0, 1000"
    mycursor.execute(sql)
    myresult = mycursor.fetchall()

    i = 0
    oritxt_list = []
    for x in myresult:
        data = SrcData(x[0], x[1], None, None, None)
        oritxt_list.append(data)

    size = len(oritxt_list)
    print("\n")
    print(str(size) + " data retrieved")

    i = 0;

    for i in tqdm(range(size)):
        oriTextObj = oritxt_list[i]
        try:
            translate_chn = translate(oriTextObj.cleanedTxt, source_lang="en", target_lang="zh-CN")
            translate_my = translate(oriTextObj.cleanedTxt, source_lang="en", target_lang="ms")
            translate_tm = translate(oriTextObj.cleanedTxt, source_lang="en", target_lang="ta")
            oriTextObj.translate_chn = translate_chn.results[0].paraphrase
            oriTextObj.translate_my = translate_my.results[0].paraphrase
            oriTextObj.translate_tm = translate_tm.results[0].paraphrase
            i = i + 1
            # print(str(i))
        except:
            print("Error at ID: " + str(oriTextObj.id))
            print("Text: " + oriTextObj.cleanedTxt)
            print("----")
    print("\n")
    print(str(len(oritxt_list)) + " data translated")
    print("\n")

    conn = sqlHelper.get_mysql_conn()
    mycursor = conn.cursor()
    sql = "UPDATE Synth_text set translate_chn = lower(%s), translate_my = lower(%s), translate_tm = lower(%s)here id = %s"

    for txtdata in oritxt_list:
        val = (txtdata.translate_chn, txtdata.translate_my, txtdata.translate_tm, txtdata.id)
        mycursor.execute(sql, val)
    conn.commit()

    # -----------------
    end_time = datetime.now()
    time_difference = (end_time - start_time).total_seconds() * 10 ** 3
    print("Execution time of program is: ", time_difference, "ms")


for exei in range(0, 1):
    execute()
    print(exei)
    time.sleep(3)
