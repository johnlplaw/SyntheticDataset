FIELD_translate_chn = "translate_chn"
FIELD_translate_my = "translate_my"
FIELD_translate_tm = "translate_tm"

FIELD_cm_en_chn = "cm_en_chn"
FIELD_cm_en_my = "cm_en_my"
FIELD_cm_en_tm = "cm_en_tm"

FIELD_cm_chn_en = "cm_chn_en"
FIELD_cm_chn_my = "cm_chn_my"
FIELD_cm_chn_tm = "cm_chn_tm"

FIELD_cm_my_en = "cm_my_en"
FIELD_cm_my_chn = "cm_my_chn"
FIELD_cm_my_tm = "cm_my_tm"

FIELD_cm_tm_en = "cm_tm_en"
FIELD_cm_tm_chn = "cm_tm_chn"
FIELD_cm_tm_my = "cm_tm_my"

FIELD_cw_en_chn = "cw_en_chn"
FIELD_cw_en_my = "cw_en_my"
FIELD_cw_en_tm = "cw_en_tm"

FIELD_cw_chn_en = "cw_chn_en"
FIELD_cw_chn_my = "cw_chn_my"
FIELD_cw_chn_tm = "cw_chn_tm"

FIELD_cw_my_en = "cw_my_en"
FIELD_cw_my_chn = "cw_my_chn"
FIELD_cw_my_tm = "cw_my_tm"

FIELD_cw_tm_en = "cw_tm_en"
FIELD_cw_tm_chn = "cw_tm_chn"
FIELD_cw_tm_my = "cw_tm_my"

field_list = [FIELD_translate_chn, FIELD_translate_my, FIELD_translate_tm,
              FIELD_cm_en_chn, FIELD_cm_en_my, FIELD_cm_en_tm,
              FIELD_cm_chn_en, FIELD_cm_chn_my, FIELD_cm_chn_tm,
              FIELD_cm_my_en, FIELD_cm_my_chn, FIELD_cm_my_tm,
              FIELD_cm_tm_en, FIELD_cm_tm_chn, FIELD_cm_tm_my,
              FIELD_cw_en_chn, FIELD_cw_en_my, FIELD_cw_en_tm,
              FIELD_cw_chn_en, FIELD_cw_chn_my, FIELD_cw_chn_tm,
              FIELD_cw_my_en, FIELD_cw_my_chn, FIELD_cw_my_tm,
              FIELD_cw_tm_en, FIELD_cw_tm_chn, FIELD_cw_tm_my
              ]


class SyncDataSet:
    id = ""
    oritxt = ""
    cleanedtxt = ""
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
                 id, oritxt, cleanedtxt,
                 translate_chn, translate_my, translate_tm,
                 cm_en_chn, cm_en_my, cm_en_tm,
                 cm_chn_en, cm_chn_my, cm_chn_tm,
                 cm_my_en, cm_my_chn, cm_my_tm,
                 cm_tm_en, cm_tm_chn, cm_tm_my,
                 cw_en_chn, cw_en_my, cw_en_tm,
                 cw_chn_en, cw_chn_my, cw_chn_tm,
                 cw_my_en, cw_my_chn, cw_my_tm,
                 cw_tm_en, cw_tm_chn, cw_tm_my):
        self.id = id
        self.oritxt = oritxt
        self.cleanedtxt = cleanedtxt

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

    def get_txt(self, fieldName):

        result = ""

        if fieldName == FIELD_translate_chn:
            result = self.translate_chn
        elif fieldName == FIELD_translate_my:
            result = self.translate_my
        elif fieldName == FIELD_translate_tm:
            result = self.translate_tm

        elif fieldName == FIELD_cm_en_chn:
            result = self.cm_en_chn
        elif fieldName == FIELD_cm_en_my:
            result = self.cm_en_my
        elif fieldName == FIELD_cm_en_tm:
            result = self.cm_en_tm

        elif fieldName == FIELD_cm_chn_en:
            result = self.cm_chn_en
        elif fieldName == FIELD_cm_chn_my:
            result = self.cm_chn_my
        elif fieldName == FIELD_cm_chn_tm:
            result = self.cm_chn_tm

        elif fieldName == FIELD_cm_my_en:
            result = self.cm_my_en
        elif fieldName == FIELD_cm_my_chn:
            result = self.cm_my_chn
        elif fieldName == FIELD_cm_my_tm:
            result = self.cm_my_tm

        elif fieldName == FIELD_cm_tm_en:
            result = self.cm_tm_en
        elif fieldName == FIELD_cm_tm_chn:
            result = self.cm_tm_chn
        elif fieldName == FIELD_cm_tm_my:
            result = self.cm_tm_my

        elif fieldName == FIELD_cw_en_chn:
            result = self.cw_en_chn
        elif fieldName == FIELD_cw_en_my:
            result = self.cw_en_my
        elif fieldName == FIELD_cw_en_tm:
            result = self.cw_en_tm

        elif fieldName == FIELD_cw_chn_en:
            result = self.cw_chn_en
        elif fieldName == FIELD_cw_chn_my:
            result = self.cw_chn_my
        elif fieldName == FIELD_cw_chn_tm:
            result = self.cw_chn_tm

        elif fieldName == FIELD_cw_my_en:
            result = self.cw_my_en
        elif fieldName == FIELD_cw_my_chn:
            result = self.cw_my_chn
        elif fieldName == FIELD_cw_my_tm:
            result = self.cw_my_tm

        elif fieldName == FIELD_cw_tm_en:
            result = self.cw_tm_en
        elif fieldName == FIELD_cw_tm_chn:
            result = self.cw_tm_chn
        elif fieldName == FIELD_cw_tm_my:
            result = self.cw_tm_my

        return result