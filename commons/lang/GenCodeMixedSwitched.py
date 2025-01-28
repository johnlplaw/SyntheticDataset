import random

import jieba
from langdetect import detect
from nltk.corpus import words as nltk_words
from translate_shell.translate import translate

Eng_code = "en"
Man_code = "zh-CN"
Ms_code = "ms"
Ta_code = "ta"


def is_english_word(word):
    # creation of this dictionary would be done outside of
    #     the function because you only need to do it once.
    dictionary = dict.fromkeys(nltk_words.words(), None)
    try:
        x = dictionary[word]
        return True
    except KeyError:
        return False


def detect_language(inputStr):
    """
    Detect the language
    :param inputStr: Input text
    :return: The language code
    """
    result = ""
    try:
        result = detect(inputStr)
        if result not in ['en', 'id', 'my', 'zh-cn', 'zh-tw', 'ta']:
            if is_english_word(inputStr):
                result = 'en'
            else:
                print("Detected " + result + ". It is not a targeted language")
                result = ""
    except:
        result = ""
    return result


def tokenization(txt, language):
    if (language == None):
        language = detect_language

    wordlist = []
    if language == Man_code:
        # print("Mand code")
        wordlist = jieba.lcut(txt)
    else:
        # print("Other code")
        wordlist = txt.split(" ")
    return wordlist


def get_random_value(from_num, to_num, count):
    numbers = list(range(from_num, to_num))
    randomList = []
    for i in range(0, count):
        while len(randomList) < count:
            rn = random.choice(numbers)
            if rn not in randomList:
                randomList.append(rn)
    return randomList


def gen_code_mixed(txt, ratio, src_lang, target_lang):
    # Get a list of a random words

    if src_lang == Man_code:
        txt = txt.replace(" ","")

    tokenised_list = tokenization(txt, src_lang)
    #print(tokenised_list)
    sentense_len = len(tokenised_list)
    random_words_cnt = round(sentense_len * (1 - ratio))
    random_value = get_random_value(0, sentense_len, random_words_cnt)

    # print("==")
    # print(random_value)
    # print("==")

    for ri in random_value:
        tobe_replaced = tokenised_list[ri - 1]
        replaced_word = translate(tobe_replaced, source_lang=src_lang, target_lang=target_lang)
        if len(replaced_word.results) > 0:
            translated_word = replaced_word.results[0].paraphrase
        else:
            # if unable to be translated, then use the original word
            translated_word = tobe_replaced
        # print(tobe_replaced + " - " + translated_word)
        tokenised_list[ri - 1] = translated_word

        # print(tokenised_list)

    return " ".join(tokenised_list)


def gen_code_switched(txt, ratio, lang1, lang2):
    tokenised_list = tokenization(txt, lang1)
    #print(tokenised_list)
    sentense_len = len(tokenised_list)
    switching_point = round(sentense_len * (ratio))
    #print(switching_point)
    lang1_txt_list = []
    lang2_txt_list = []

    for i in range(0, switching_point):
        tmp = tokenised_list[i]
        lang1_txt_list.append(tmp)

    for i in range(switching_point, sentense_len):
        tmp = tokenised_list[i]
        lang2_txt_list.append(tmp)

    #print(lang1_txt_list)
    #print(lang2_txt_list)

    lang1_txt = " ".join(lang1_txt_list)
    lang2_txt = " ".join(lang2_txt_list)
    translates_txt = translate(lang2_txt, source_lang=lang1, target_lang=lang2)

    translated_txt = translates_txt.results[0].paraphrase

    # handling mandarin tokenization
    if lang2 == Man_code:
        translated_txt_tokens = tokenization(translated_txt, lang2)
        translated_txt = " ".join(translated_txt_tokens)

    return lang1_txt + " " + translated_txt
