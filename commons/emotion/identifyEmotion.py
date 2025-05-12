import text2emotion as te

"""
By default the text2emotion will get error, AttributeError: module 'emoji' has no attribute 'UNICODE_EMOJI'
By referring to https://www.datasciencelearner.com/python-exceptions/attributeerror/attributeerror-module-emoji-has-no-attribute-unicode-emoji-solved/

Update the library:
pip uninstall emoji
pip install emoji==1.7.0
"""


def identifyEmotion(input):
    emotions = te.get_emotion(input)

    if (max(emotions.values()) == 0):
        return "Neutral"
    else:
        return max(emotions, key=emotions.get)


# # txt = "#johor"
# txt = "Scare Time to get start... yeah.."
# emo = identifyEmotion(txt)
# print(emo)
