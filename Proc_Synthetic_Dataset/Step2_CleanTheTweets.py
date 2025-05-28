# This script is for cleaning the English text.
# 2024-01-07 - Initial Coding

import commons.lang.CleanText as ct
import commons.mysql.mysqlHelper as sqlHelper


def light_clean(input):
    # not include the stop word and punctuation removal since they still have impact to the BERT
    txt = ct.removebStr(input)
    txt = ct.removeURL(txt)
    txt = ct.remove_a_tag(txt)
    txt = ct.remove_html_tag(txt)
    txt = ct.removeWordStarsWithChar(txt)
    txt = ct.transform_emojis_into_char(txt)
    txt = ct.removeChar(txt)

    txt = ct.remove_repeated_char(txt)
    txt = ct.remove_redundant_space(txt)

    txt = ct.replaceChar(txt)
    txt = ct.toLowerCase(txt)
    txt = ct.replace_numeric_with_symbol(txt)
    # txt = ct.spelling_correction(txt)
    return txt


class SrcData:
    id = ""
    oriTxt = ""
    labelTxt = ""
    cleanedTxt = ""

    def __init__(self, id, oriTxt, labelTxt, cleanedTxt):
        self.id = id
        self.oriTxt = oriTxt
        self.labelTxt = labelTxt
        self.cleanedTxt = cleanedTxt

    def __str__(self):
        return str(self.id) + " | " + self.oriTxt + " | " + self.labelTxt + " | " + str(self.cleanedTxt)


# Get the oritxt from mysql
conn = sqlHelper.get_mysql_conn()

mycursor = conn.cursor()
sql = "SELECT ID, label, oritxt, cleanedtxt from Synth_text where cleanedtxt is null "
mycursor.execute(sql)
myresult = mycursor.fetchall()

i = 0
oritxt_list = []
for x in myresult:
    data = SrcData(x[0], x[2], x[1], x[3])
    oritxt_list.append(data)

print(str(len(oritxt_list)) + " data retrieved")

for dt in oritxt_list:
    dt.cleanedTxt = light_clean(dt.oriTxt)

print(str(len(oritxt_list)) + " data processed")

print("Update into mysql - start")

conn = sqlHelper.get_mysql_conn()
mycursor = conn.cursor()
sql = "UPDATE Synth_text set cleanedtxt = %s where id = %s"

for txtdata in oritxt_list:
    val = (txtdata.cleanedTxt, txtdata.id)
    mycursor.execute(sql, val)

conn.commit()
print(mycursor.rowcount, "record updated.")

print("Update into mysql - end")
