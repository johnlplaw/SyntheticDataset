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
sql = "SELECT ID, label, oritxt, cleanedtxt from mydataset where cleanedtxt is null "
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
sql = "UPDATE mydataset set cleanedtxt = %s where id = %s"

for txtdata in oritxt_list:
    val = (txtdata.cleanedTxt, txtdata.id)
    mycursor.execute(sql, val)

conn.commit()
print(mycursor.rowcount, "record updated.")

print("Update into mysql - end")

# Backup:
# ---------------
# wb = openpyxl.load_workbook("./data/SourceDataSets.xlsx")
# sheets = wb.sheetnames


# ws = wb[sheets[3]]
# column1 = ws['A']
# column2 = ws['B']
# column3 = ws['C']


# x = ""
# i = 0
# for row in range(1, ws.max_row):
# for row in range(11, 1000):
#     labelTxt = str(column1[row].value)
#     oriTxt = str(column3[row].value)
#     cleanedTxt = str(column2[row].value)
#     mycleanTxt = str(light_clean(oriTxt))
#
#     print("orig: " + oriTxt)
#     print("exam: " + cleanedTxt)
#     print("____: " + mycleanTxt)
#     print("========================")
#     x = x + "\norig: " + oriTxt
#     x = x + "\nexam: " + cleanedTxt
#     x = x + "\n____: " + mycleanTxt
#     x = x + "\n-------------"
#
# with open("new.txt","w") as f:
#   f.writelines(x)
#   f.close()
