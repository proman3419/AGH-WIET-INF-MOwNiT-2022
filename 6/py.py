from Entry import Entry
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

# print(num2words(1923))
# print(re.sub(r'[^a-zA-Z\d\s:]', '', 
#                                        num2words(13334)))

import pickle

# Entry    id: int
#     url: str
#     title: str
#     text: str
#     words_set: set
#     words_vec: List[int]
#     additional_info: dict
# e1 = Entry(0, 'google.com', 'wyszukiwarka and stuff', 'siema', {'wyszukiwarka': 1, 'stuff': 1}, [1, 1], additional_info={})
# e2 = Entry(1, 'wiki.com', 'wyszukiwarka and sadf', 'no hej', {'wyszukiwarka': 1, 'stuff': 1}, [1, 1, 1], additional_info={})
# dic = {1: 'asdfafd', 'dsaf': 'asfdadsfs'}

# with open('out.pkl', 'wb+') as f:
#     pickle.dump(e1, f)
#     pickle.dump(e2, f)
#     pickle.dump(dic, f)

# del e1
# del e2
# del dic

# with open('out.pkl', 'rb') as f:
#     e1 = pickle.load(f)
#     e2 = pickle.load(f)
#     dic = pickle.load(f)

# print(e1)
# print(e2)
# print(dic)


# with open('adsfdasfas', 'r') as f:
#     print(f)

from collections import defaultdict

a = defaultdict(int)
# print(type(a.keys() | {1}))

from scipy.sparse import csc_matrix
import numpy as np

data = np.array([1, 2, 3, 4, 5, 6])
print(csc_matrix(data).toarray())
