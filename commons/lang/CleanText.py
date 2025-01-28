import itertools
import re
from textblob import TextBlob
import demoji
import csv

# 1. Remove b'...' from the text
def removebStr(input):

    charForTxt = ['\'', '"']

    for txtChar in charForTxt:

        if input.startswith("b" + txtChar):
            input = input.replace("b" + txtChar, "")
            if input.endswith(txtChar):
                input = input[:-1]

    return input

# 2. Remove the URL
def removeURL(input):
    input = input + " "  # to handle the text ends with a URL
    input = re.sub(r'http\S+', "", input)
    return input.strip()

# 3. Remove <a> tags
def remove_a_tag(input):
    return re.sub(r"<a[^>]*>(.*?)</a>", r"\1", input)

# 4. remove html tag and its contents
def remove_html_tag(input):
    return re.sub(r"<.*?>", " ", input)

# 5. Remove the words start with @
def removeWordStarsWithChar(input):
    charList = ['@', '#']
    for theChar in charList:
        input = re.sub(theChar + ".*? ", "", input)
    return input

# 6. transform the emojis into characters
def transform_emojis_into_char(input):
    result_dic = demoji.findall(input)
    inputList = input.split()
    newInputList = []
    for a in inputList:
        if result_dic.get(a):
            emoji_desc = result_dic.get(a)
            emoji_descList = emoji_desc.split(":")
            a = emoji_descList[0].replace(" ", "_").strip()
        newInputList.append(a)
    return " ".join(newInputList)

# 7. Remove the special chars
def removeChar(input):
    removed_chars = ['RT', '*', '>']
    for removed_char in removed_chars:
        input = input.replace(removed_char, "")

    return input

# 8. Remove the repeated chars at the end of the word
def remove_repeated_char(input):
    #return ''.join(ch for ch, _ in itertools.groupby(input))
    #return re.sub(r'(.)\1{2,}',r'\1', input)
    return re.sub(r'(.)\1{3,}',r'\1', input)

# 9. Remove redundant blank space
def remove_redundant_space(input):
    # replace multiple spaces with a single space
    result = " ".join(input.split())
    return result

# 10. Replace the special chars with empty space
def replaceChar(input):
    replaced_chars = ['\\n',':']
    for replaced_char in replaced_chars:
        input = input.replace(replaced_char, " ")

    return input

# 11. Turn the text into lower case
def toLowerCase(input):
    input = input.lower()
    return input

# 12. Replace numerical value with a symbol
def replace_numeric_with_symbol (input):
    words_list = input.split()
    newWord_list = []
    for theword in words_list:
        if theword.isdigit():
            theword = 'num'
        newWord_list.append(theword)

    return " ".join(newWord_list)


    clean_text = " ".join([w for w in input.split() if not w.isdigit()])
    return ""

# 13. remove # word
def remove_hashWord (input):
    return ' '.join(word for word in input.split(' ') if not word.startswith('#'))

def get_shortform_list(shortform_csv_path):
    """
    # The malay short form list is taken from:
    # 1. https://cilisos.my/bahasa-sms-shortforms-glossary/
    # 2. https://www.kaggle.com/datasets/dennisherdi/indonesian-twitter-emotion?select=kamus_singkatan.csv

    :return:
    """
    shortform_list = []

    with open(shortform_csv_path, newline='') as csvfile:
        list = csv.reader(csvfile, delimiter=';')
        for item in list:
            shortform_list.append(item)

    return shortform_list

def replace_malay_shortform(input, shortform_list):
    """

    :param input: A text with Malay short form
    :return: A text has been replaced the short form
    """
    strList = input.split(" ")
    replacedStrList = []
    for str in strList:
        foundTxt = ""
        for row in shortform_list:
            if row[0] == str:
                print("found")
                foundTxt = row[1]

        if foundTxt != "":
            replacedStrList.append(foundTxt)
        else:
            replacedStrList.append(str)

    output = " ".join(replacedStrList)
    return output

