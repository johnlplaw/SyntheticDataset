import commons.lang.GenCodeMixedSwitched as cm
import commons.mysql.mysqlHelper as sqlHelper
from datetime import datetime
from tqdm import tqdm
import time

# txt_Eng = "i feel that its in that trusting him despite what we see around us that we really see gods glory shine"
# txt_Man = "我觉得无论我们周围看到什么，只要相信他，我们就能真正看到神的荣耀闪耀"
# txt_ms = "Saya merasakan bahawa dengan mempercayainya walaupun apa yang kita lihat di sekeliling kita bahawa kita benar-benar melihat kemuliaan tuhan bersinar"
# txt_ta = "நாம் நம்மைச் சுற்றி என்ன பார்த்தாலும் அவர் மீது நம்பிக்கை வைப்பதில்தான் கடவுளின் மகிமை பிரகாசிப்பதைக் காண்கிறோம் என்று நான் உணர்கிறேன்"

# print(gen_code_mixed(txt_Eng, 0.5, cm.Eng_code, cm.Man_code))
# print(gen_code_mixed(txt_Eng, 0.5, cm.Eng_code, cm.Ms_code))
# print(gen_code_mixed(txt_Eng, 0.5, cm.Eng_code, cm.Ta_code))
#
# print(gen_code_mixed(txt_Man, 0.5, cm.Man_code, cm.Eng_code))
# print(gen_code_mixed(txt_Man, 0.5, cm.Man_code, cm.Ms_code))
# print(gen_code_mixed(txt_Man, 0.5, cm.Man_code, cm.Ta_code))
#
# print(gen_code_mixed(txt_ms, 0.5, cm.Ms_code, cm.Man_code))
# print(gen_code_mixed(txt_ms, 0.5, cm.Ms_code, cm.Eng_code))
# print(gen_code_mixed(txt_ms, 0.5, cm.Ms_code, cm.Ta_code))
#
# print(gen_code_mixed(txt_ta, 0.5, cm.Ta_code, cm.Man_code))
# print(gen_code_mixed(txt_ta, 0.5, cm.Ta_code, cm.Eng_code))
# print(gen_code_mixed(txt_ta, 0.5, cm.Ta_code, cm.Ms_code))

# ---------
start_time = datetime.now()


class SrcData:
    id = ""
    cleanedTxt = ""
    translate_chn = ""
    translate_my = ""
    translate_tm = ""

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

    def __init__(self, id, cleanedTxt, translate_chn, translate_my, translate_tm,
                 cw_en_chn, cw_en_my, cw_en_tm,
                 cw_chn_en, cw_chn_my, cw_chn_tm,
                 cw_my_en, cw_my_chn, cw_my_tm,
                 cw_tm_en, cw_tm_chn, cw_tm_my):
        self.id = id
        self.cleanedTxt = cleanedTxt
        self.translate_chn = translate_chn
        self.translate_my = translate_my
        self.translate_tm = translate_tm

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

    def __str__(self):
        return str(self.id) + " | " + self.cleanedTxt + " | " + self.translate_chn + " | " + str(
            self.translate_my) + " | " + str(self.translate_tm)


def execute():
    # Get the oritxt from mysql
    conn = sqlHelper.get_mysql_conn()

    mycursor = conn.cursor()
    sql = ("select " +
           "id, cleanedTxt, translate_chn, translate_my, translate_tm " +
           "from mydataset "
           "where "
           " not ("
           " cleanedTxt = '' or cleanedTxt is null "
           " ) and ( "
           " cw_en_chn is null or "
           " cw_en_my  is null or "
           " cw_en_tm  is null or "
           " cw_chn_en  is null or "
           " cw_chn_my  is null or "
           " cw_chn_tm  is null or "
           " cw_my_en  is null or "
           " cw_my_chn  is null or "
           " cw_my_tm  is null or "
           " cw_tm_en is null or "
           " cw_tm_chn is null or "
           " cw_tm_my  is null ) "
           "limit 0, 50")
    mycursor.execute(sql)
    myresult = mycursor.fetchall()

    i = 0
    oritxt_list = []
    for x in myresult:
        data = SrcData(x[0], x[1], x[2], x[3], x[4],
                       None, None, None,
                       None, None, None,
                       None, None, None,
                       None, None, None)
        oritxt_list.append(data)

    size = len(oritxt_list)
    print("\n")
    print(str(size) + " data retrieved")

    i = 0;
    for i in tqdm(range(size)):
        oriTextObj = oritxt_list[i]
        try:
            cw_en_chn = cm.gen_code_switched(oriTextObj.cleanedTxt, 0.5, cm.Eng_code, cm.Man_code)
            cw_en_my = cm.gen_code_switched(oriTextObj.cleanedTxt, 0.5, cm.Eng_code, cm.Ms_code)
            cw_en_tm = cm.gen_code_switched(oriTextObj.cleanedTxt, 0.5, cm.Eng_code, cm.Ta_code)

            # print(oriTextObj.cleanedTxt)
            # print(cw_en_chn)
            # print(cw_en_my)
            # print(cw_en_tm)

            cw_chn_en = cm.gen_code_switched(oriTextObj.translate_chn, 0.5, cm.Man_code, cm.Eng_code)
            cw_chn_my = cm.gen_code_switched(oriTextObj.translate_chn, 0.5, cm.Man_code, cm.Ms_code)
            cw_chn_tm = cm.gen_code_switched(oriTextObj.translate_chn, 0.5, cm.Man_code, cm.Ta_code)

            cw_my_en = cm.gen_code_switched(oriTextObj.translate_my, 0.5, cm.Ms_code, cm.Eng_code)
            cw_my_chn = cm.gen_code_switched(oriTextObj.translate_my, 0.5, cm.Ms_code, cm.Man_code)
            cw_my_tm = cm.gen_code_switched(oriTextObj.translate_my, 0.5, cm.Ms_code, cm.Ta_code)

            cw_tm_en = cm.gen_code_switched(oriTextObj.translate_tm, 0.5, cm.Ta_code, cm.Eng_code)
            cw_tm_chn = cm.gen_code_switched(oriTextObj.translate_tm, 0.5, cm.Ta_code, cm.Man_code)
            cw_tm_my = cm.gen_code_switched(oriTextObj.translate_tm, 0.5, cm.Ta_code, cm.Ms_code)

            oriTextObj.cw_en_chn = cw_en_chn
            oriTextObj.cw_en_my = cw_en_my
            oriTextObj.cw_en_tm = cw_en_tm

            oriTextObj.cw_chn_en = cw_chn_en
            oriTextObj.cw_chn_my = cw_chn_my
            oriTextObj.cw_chn_tm = cw_chn_tm

            oriTextObj.cw_my_en = cw_my_en
            oriTextObj.cw_my_chn = cw_my_chn
            oriTextObj.cw_my_tm = cw_my_tm

            oriTextObj.cw_tm_en = cw_tm_en
            oriTextObj.cw_tm_chn = cw_tm_chn
            oriTextObj.cw_tm_my = cw_tm_my

        except:
            print("Error at ID: " + str(oriTextObj.id))
            print("Text: " + oriTextObj.cleanedTxt)
            print("----")
    print(str(len(oritxt_list)) + " data translated")

    conn = sqlHelper.get_mysql_conn()
    mycursor = conn.cursor()
    sql = ("UPDATE mydataset set" +
           " cw_en_chn = lower(%s), cw_en_my = lower(%s), cw_en_tm = lower(%s)," +
           " cw_chn_en = lower(%s), cw_chn_my = lower(%s), cw_chn_tm = lower(%s)," +
           " cw_my_en = lower(%s), cw_my_chn = lower(%s), cw_my_tm = lower(%s)," +
           " cw_tm_en = lower(%s), cw_tm_chn = lower(%s), cw_tm_my = lower(%s)" +
           " where id = %s")

    for oriTextObj in oritxt_list:
        val = (oriTextObj.cw_en_chn, oriTextObj.cw_en_my, oriTextObj.cw_en_tm,
               oriTextObj.cw_chn_en, oriTextObj.cw_chn_my, oriTextObj.cw_chn_tm,
               oriTextObj.cw_my_en, oriTextObj.cw_my_chn, oriTextObj.cw_my_tm,
               oriTextObj.cw_tm_en, oriTextObj.cw_tm_chn, oriTextObj.cw_tm_my,
               oriTextObj.id)
        mycursor.execute(sql, val)
    conn.commit()

    # -----------------
    end_time = datetime.now()
    time_difference = (end_time - start_time).total_seconds() * 10**3
    print("Execution time of program is: ", time_difference, "ms")


for exei in range(0, 1000):
    execute()
    print(exei)
    time.sleep(3)
