# import openai
# from nltk.tokenize import sent_tokenize
# import nltk 
# nltk.download('punkt')

# openai.api_key = "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"

from answer import getAnswer

myQuestion = "What is your name?"

ans = getAnswer(myQuestion)

print(ans)