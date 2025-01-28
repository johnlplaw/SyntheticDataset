from rippletagger.tagger import Tagger


def convert_lang_code_for_tagging(language):
    # 'en', 'id', 'my', 'zh-cn', 'zh-tw', 'ta']
    converted_code = ""
    if language == "en":
        converted_code = "en"
    elif language in ["id","my"]:
        converted_code = "id"
    elif language in ["zh-cn", "zh-tw"]:
        converted_code = "zh"
    elif language in "ta":
        converted_code = "tam"
    return converted_code


def pos_tagging(language, input_text):

    language_code = convert_lang_code_for_tagging (language)

    tagger = Tagger(language=language_code)
    pos_tagger = tagger.tag(input_text)
    return pos_tagger