import re


text = '''Lorem Ipsum is simply dummy text of the printing and typesetting industry. 

    Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a          type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. 


    It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.'''

text = text.lower() # małe litery
text = re.sub(r'[^a-zA-Z\d\s:]','',text) # usunięcie znaków niealfanumerycznych
text = re.sub(r'\s\s+', ' ',text) # usunięcie zbędnych białych znaków
# print(text)

from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
# print(lemmatizer.lemmatize("rocks", pos='n'))

from num2words import num2words

print(num2words(1923))
print(re.sub(r'[^a-zA-Z\d\s:]', '', 
                                       num2words(13334)))
