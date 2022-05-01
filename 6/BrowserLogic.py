from Entry import Entry
from abc import ABC, abstractmethod
from typing import List
import nltk
from nltk.stem import WordNetLemmatizer
from num2words import num2words
import re


class BrowserLogic(ABC):
    def __init_subclass__(cls, **kwargs):
        cls.entries: List[Entry] = []
        cls.dictionary: List[str] = []

    def fit(self):
        self.load_entries()
        self.preprocess_entries()
        self.generate_dictionary()

    @abstractmethod
    def load_entries(self):
        raise NotImplementedError

    def preprocess_entries(self):
        def preprocess(text):
            text = text.lower() # małe litery
            text = re.sub(r'[^a-zA-Z\d\s:]', '', text) # usunięcie znaków niealfanumerycznych
            text = re.sub(r'\s\s+', ' ', text) # usunięcie zbędnych białych znaków
            return text

        nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer() 

        for entry in self.entries:
            entry.text = preprocess(entry.text)
            for word in entry.text.split():
                try:
                    # zamiana liczby na słowną reprezentację
                    sub_words = preprocess(num2words(int(word)))
                    for sub_word in sub_words:
                        entry.words_set.add(sub_word)
                except ValueError:
                    # sprowadzenie słowa do podstawowej formy
                    _word = lemmatizer.lemmatize(word, pos='n')
                    if _word == word: _word = lemmatizer.lemmatize(word, pos='v')
                    if _word == word: _word = lemmatizer.lemmatize(word, pos='a')
                    word = _word

                    entry.words_set.add(preprocess(word))

    def generate_dictionary(self):
        self.dictionary = set()
        for entry in self.entries:
            self.dictionary |= entry.words_set
