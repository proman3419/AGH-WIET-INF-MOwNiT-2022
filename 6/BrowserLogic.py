from Entry import Entry
import const
from abc import ABC, abstractmethod
from typing import List, Set
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import pickle


class BrowserLogic(ABC):
    def __init_subclass__(cls, **kwargs):
        cls.entries: List[Entry] = []
        cls.dictionary: Set(str) = set()
        nltk.download('wordnet')
        nltk.download('stopwords')

    def fit(self, _load_dumped=False):
        loaded = False
        if _load_dumped:
            try:
                self.entries = self.load_dumped(const.DUMP_FP_ENTRIES)
                self.dictionary = self.load_dumped(const.DUMP_FP_DICTIONARY)
                print('Loaded dumped files')
                loaded = True
            except FileNotFoundError:
                pass
        if not loaded:
            self.init_entries()
            self.preprocess_entries()

    @abstractmethod
    def init_entries(self):
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
                    entry.words_dict[_word] += 1

            return entry.words_dict.keys()

        stopwords_en = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        self.dictionary = set()
        for entry in self.entries:
            self.dictionary |= preprocess(entry)

        self.dump(const.DUMP_FP_ENTRIES, self.entries)
        self.dump(const.DUMP_FP_DICTIONARY, self.dictionary)

    def dump(self, dump_fp, obj):
        with open(dump_fp, 'wb+') as dump_f:
            pickle.dump(obj, dump_f)

    def load_dumped(self, dump_fp):
        with open(dump_fp, 'rb') as dump_f:
            obj = pickle.load(dump_f)
        return obj
