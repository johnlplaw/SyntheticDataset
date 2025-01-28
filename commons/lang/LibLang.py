import re

import jieba
from googletrans import Translator
from langdetect import detect
from nltk.corpus import words as nltk_words
from translate_shell.translate import translate

### python3 -m pip install googletrans==3.1.0.a0
### https://github.com/uliontse/translators
### https://stackabuse.com/text-translation-with-google-translate-api-in-python/
### https://translate-shell.readthedocs.io/en/latest/
###

symbols_list = [',','，','!','！','.','。']

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


def translatePrc(oriText, fromLang, toLang):
    """
    Translate the language to the desired language
    :param oriText: The original text
    :param fromLang: From language code
    :param toLang: To language code
    :return: Translated text
    """
    translator = Translator()
    return translator.translate(oriText, dest=toLang, src=fromLang)


def is_chinese_text_found(inputText):
    """
    Check whether there is any Chinese character found
    :param inputText: The input text
    :return: True if there is, false otherwise
    """
    chnWord = re.findall(u'[\u4e00-\u9fff]+', inputText)
    return len(chnWord) > 0


def is_only_chinese_text(inputText):
    """
    Check whether it is pure chinese text
    :param inputText: The input text
    :return: Ture if there is, false otherwise
    """

    wordList = inputText.split(" ")
    otherLangDetected = 0
    for w in wordList:
        chnWord = re.findall(u'[\u4e00-\u9fff]+', w)

        if len(chnWord) == 0:
            otherLangDetected = otherLangDetected + 1

    return otherLangDetected == 0


def sigment_chn_text(inputText):
    """
    Segmentation of the Chinese text
    :param inputText: The input text
    :return: return a list containing the segmented text
    """
    return jieba.lcut(inputText)


def extract_chn_text(inputText):
    """
    Extract the Chinese words (u4e00 to u9fff)
    :param inputText: The input text
    :return: A List of Chinese words
    """
    return re.findall(u'[\u4e00-\u9fff]+', inputText)


def is_tamil_text_found(inputText):
    """
    To check whether the input text has any Tamil text
    :param inputText: The input text
    :return: True if there is Tamil word, false otherwise
    """
    tmlWord = re.findall(u'[\u0B82-\u0BFA]+', inputText)
    return len(tmlWord) > 0


def is_only_tamil_text(inputText):
    """
    To check whether the input text is tamil
    :param inputText: The input text
    :return: True it is Tamil text, false otherwise
    """
    wordList = inputText.split(" ")
    otherLangDetected = 0
    for w in wordList:
        chnWord = re.findall(u'[\u0B82-\u0BFA]+', w)

        if len(chnWord) == 0 :
            otherLangDetected = otherLangDetected + 1

    return otherLangDetected == 0


def sigment_tamil_text(inputText):
    """
    Segmentation of the Tamil text
    :param inputText:
    :return:
    """
    return inputText.split(" ")


def extract_tamil_words(inputTxt):
    """
    Extract the tamil words
    :param inputTxt:
    :return:
    """
    return re.findall(u'[\u0B82-\u0BFA]+', inputTxt)


def is_chn_char(input):
    """

    :param input:
    :return:
    """
    chnWord = re.findall(u'[\u4e00-\u9fff]+', input)
    return len(chnWord) > 0

def is_tamil_char(input):
    tamWord = re.findall(u'[\u0B82-\u0BFA]+', input)
    return len(tamWord) > 0

def fix_space_multilingual (input):
    fix_input = ""
    prevChar = ""
    prevLang = ""
    for curr_word in input:
        lang = ""
        if is_chn_char(curr_word):
            lang = "cn"
        if is_tamil_char(curr_word):
            lang = "tam"

        if prevLang != lang and curr_word not in symbols_list and curr_word != " " and prevChar != " ":
            fix_input = fix_input + " " + curr_word
        else :
            fix_input = fix_input + curr_word

        prevChar = curr_word
        prevLang = lang
    return fix_input


def convertTradToSimplChn(inputText):
    translated = translate(inputText, source_lang="zh_TW", target_lang="zh-CN")
    translatedTxt = translated.results[0]
    print(type(translatedTxt))

    return translatedTxt["paraphrase"]

