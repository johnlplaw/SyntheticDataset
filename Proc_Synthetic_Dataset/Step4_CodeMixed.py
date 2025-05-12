import commons.lang.GenCodeMixedSwitched as cm
import commons.mysql.mysqlHelper as sqlHelper
from datetime import datetime
from tqdm import tqdm
import time

# Initial Coding - start
#########################
# txt_Eng = "i feel that its in that trusting him despite what we see around us that we really see gods glory shine"
# txt_Man = "我觉得无论我们周围看到什么，只要相信他，我们就能真正看到神的荣耀闪耀"
# txt_ms = "Saya merasakan bahawa dengan mempercayainya walaupun apa yang kita lihat di sekeliling kita bahawa kita benar-benar melihat kemuliaan tuhan bersinar"
# txt_ta = "நாம் நம்மைச் சுற்றி என்ன பார்த்தாலும் அவர் மீது நம்பிக்கை வைப்பதில்தான் கடவுளின் மகிமை பிரகாசிப்பதைக் காண்கிறோம் என்று நான் உணர்கிறேன்"
#
# translate_chn = translate(txt, source_lang="en", target_lang="zh-CN")
# translate_my = translate(txt, source_lang="en", target_lang="ms")
# translate_tm = translate(txt, source_lang="en", target_lang="ta")
#
# translate_chn = translate_chn.results[0].paraphrase
# translate_my = translate_my.results[0].paraphrase
# translate_tm = translate_tm.results[0].paraphrase
#
#
# print(translate_chn)
# print(translate_my)
# print(translate_tm)

# print(gen_code_mixed(txt_Eng, 0.8, Eng_code, Man_code))
# print(gen_code_mixed(txt_Eng, 0.8, Eng_code, Ms_code))
# print(gen_code_mixed(txt_Eng, 0.8, Eng_code, Ta_code))
#
# print(gen_code_mixed(txt_Man, 0.8, Man_code, Eng_code ))
# print(gen_code_mixed(txt_Man, 0.8, Man_code, Ms_code ))
# print(gen_code_mixed(txt_Man, 0.8, Man_code, Ta_code ))
#
# print(gen_code_mixed(txt_ms, 0.8, Ms_code, Eng_code ))
# print(gen_code_mixed(txt_ms, 0.8, Ms_code, Man_code ))
# print(gen_code_mixed(txt_ms, 0.8, Ms_code, Ta_code ))
#
# print(gen_code_mixed(txt_ta, 0.8, Ta_code, Eng_code ))
# print(gen_code_mixed(txt_ta, 0.8, Ta_code, Man_code ))
# print(gen_code_mixed(txt_ta, 0.8, Ta_code, Ms_code ))
# Initial Coding - end
#########################


start_time = datetime.now()


# -----------------

# ==================================

class SrcData:
    id = ""
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

    def __init__(self, id, cleanedTxt, translate_chn, translate_my, translate_tm,
                 cm_en_chn, cm_en_my, cm_en_tm,
                 cm_chn_en, cm_chn_my, cm_chn_tm,
                 cm_my_en, cm_my_chn, cm_my_tm,
                 cm_tm_en, cm_tm_chn, cm_tm_my):
        self.id = id
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

    def __str__(self):
        return str(self.id) + " | " + self.cleanedTxt + " | " + self.translate_chn + " | " + str(
            self.translate_my) + " | " + str(self.translate_tm)

def execute():
    # Get the oritxt from mysql
    conn = sqlHelper.get_mysql_conn()

    mycursor = conn.cursor()
    sql = ("select " +
           "id, cleanedTxt, translate_chn, translate_my, translate_tm " +
           "from Synth_text "
           "where "
           " not ("
           " cleanedTxt = '' or cleanedTxt is null "
           " ) and "
           "("
           " translate_chn is not null "
           " and translate_my is not null "
           " and translate_tm is not null "
           ") and ( "
           " cm_en_chn is null or "
           " cm_en_my  is null or "
           " cm_en_tm  is null or "
           " cm_chn_en  is null or "
           " cm_chn_my  is null or "
           " cm_chn_tm  is null or "
           " cm_my_en  is null or "
           " cm_my_chn  is null or "
           " cm_my_tm  is null or "
           " cm_tm_en is null or "
           " cm_tm_chn is null or "
           " cm_tm_my  is null ) "
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
            cm_en_chn = cm.gen_code_mixed(oriTextObj.cleanedTxt, 0.8, cm.Eng_code, cm.Man_code)
            cm_en_my = cm.gen_code_mixed(oriTextObj.cleanedTxt, 0.8, cm.Eng_code, cm.Ms_code)
            cm_en_tm = cm.gen_code_mixed(oriTextObj.cleanedTxt, 0.8, cm.Eng_code, cm.Ta_code)

            #print(oriTextObj.translate_chn)

            cm_chn_en = cm.gen_code_mixed(oriTextObj.translate_chn, 0.8, cm.Man_code, cm.Eng_code)
            cm_chn_my = cm.gen_code_mixed(oriTextObj.translate_chn, 0.8, cm.Man_code, cm.Ms_code)
            cm_chn_tm = cm.gen_code_mixed(oriTextObj.translate_chn, 0.8, cm.Man_code, cm.Ta_code)



            cm_my_en = cm.gen_code_mixed(oriTextObj.translate_my, 0.8, cm.Ms_code, cm.Eng_code)
            cm_my_chn = cm.gen_code_mixed(oriTextObj.translate_my, 0.8, cm.Ms_code, cm.Man_code)
            cm_my_tm = cm.gen_code_mixed(oriTextObj.translate_my, 0.8, cm.Ms_code, cm.Ta_code)

            cm_tm_en = cm.gen_code_mixed(oriTextObj.translate_tm, 0.8, cm.Ta_code, cm.Eng_code)
            cm_tm_chn = cm.gen_code_mixed(oriTextObj.translate_tm, 0.8, cm.Ta_code, cm.Man_code)
            cm_tm_my = cm.gen_code_mixed(oriTextObj.translate_tm, 0.8, cm.Ta_code, cm.Ms_code)

            oriTextObj.cm_en_chn = cm_en_chn
            oriTextObj.cm_en_my = cm_en_my
            oriTextObj.cm_en_tm = cm_en_tm

            oriTextObj.cm_chn_en = cm_chn_en
            oriTextObj.cm_chn_my = cm_chn_my
            oriTextObj.cm_chn_tm = cm_chn_tm

            oriTextObj.cm_my_en = cm_my_en
            oriTextObj.cm_my_chn = cm_my_chn
            oriTextObj.cm_my_tm = cm_my_tm

            oriTextObj.cm_tm_en = cm_tm_en
            oriTextObj.cm_tm_chn = cm_tm_chn
            oriTextObj.cm_tm_my = cm_tm_my

        except Exception as error:
            print("Error at ID: " + str(oriTextObj.id))
            print("Text: " + oriTextObj.cleanedTxt)
            print("An exception occurred:", error)
            print("----")
    print(str(len(oritxt_list)) + " data translated")

    conn = sqlHelper.get_mysql_conn()
    mycursor = conn.cursor()
    sql = ("UPDATE Synth_text set" +
           " cm_en_chn = lower(%s), cm_en_my = lower(%s), cm_en_tm = lower(%s)," +
           " cm_chn_en = lower(%s), cm_chn_my = lower(%s), cm_chn_tm = lower(%s)," +
           " cm_my_en = lower(%s), cm_my_chn = lower(%s), cm_my_tm = lower(%s)," +
           " cm_tm_en = lower(%s), cm_tm_chn = lower(%s), cm_tm_my = lower(%s)" +
           " where id = lower(%s)")

    for oriTextObj in oritxt_list:
        val = (oriTextObj.cm_en_chn, oriTextObj.cm_en_my, oriTextObj.cm_en_tm,
               oriTextObj.cm_chn_en, oriTextObj.cm_chn_my, oriTextObj.cm_chn_tm,
               oriTextObj.cm_my_en, oriTextObj.cm_my_chn, oriTextObj.cm_my_tm,
               oriTextObj.cm_tm_en, oriTextObj.cm_tm_chn, oriTextObj.cm_tm_my,
               oriTextObj.id)
        mycursor.execute(sql, val)
    conn.commit()

    # -----------------
    end_time = datetime.now()
    time_difference = (end_time - start_time).total_seconds() * 10 ** 3
    print("Execution time of program is: ", time_difference, "ms")

for exei in range(0, 1000):
    execute()
    print(exei)
    time.sleep(3)
