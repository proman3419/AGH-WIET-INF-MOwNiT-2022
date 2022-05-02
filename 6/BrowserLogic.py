from Entry import Entry
from abc import ABC, abstractmethod
from typing import List
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import tensorflow as tf


class BrowserLogic(ABC):
    def __init_subclass__(cls, **kwargs):
        cls.entries: List[Entry] = []
        cls.dictionary: List[str] = []

    def fit(self):
        self.load_entries()
        words = self.preprocess_entries()
        self.generate_dictionary(words)

    @abstractmethod
    def load_entries(self):
        raise NotImplementedError

    def preprocess_entries(self):
        def preprocess(entry):
            nonlocal lemmatizer, stopwords_en
            text = entry.text
            text = text.lower() # małe litery
            text = re.sub(r'["`\'()\[\]{}|,.:;!?\-_–]+', ' ', text) # znaki nieniosące treści
            text = re.sub(r'\s\s+', ' ', text) # usunięcie zbędnych białych znaków
            for word in text.split():
                _word = lemmatizer.lemmatize(word, pos='v')
                if _word == word: _word = lemmatizer.lemmatize(word, pos='n')
                if _word == word: _word = lemmatizer.lemmatize(word, pos='a')
                if _word not in stopwords_en:
                    entry.words_set.add(_word)

            return entry.words_set

        nltk.download('wordnet')
        stopwords_en = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        words = set()
        for entry in self.entries:
            words |= preprocess(entry)

        return words


    def generate_dictionary(self, words):
        print(words)
        tokenizer = tf.keras.preprocessing.text.Tokenizer(list(words))
        tokenizer.fit_on_texts(words)
        print(tokenizer.word_counts)
        print(len(tokenizer.word_counts))
