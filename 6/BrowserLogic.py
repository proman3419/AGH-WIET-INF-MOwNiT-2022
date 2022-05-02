from Entry import Entry
import const
from abc import ABC, abstractmethod
from typing import List, Dict
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import pickle
from collections import defaultdict
from scipy.sparse import csr_matrix
import numpy as np


class BrowserLogic(ABC):
    def __init_subclass__(cls, **kwargs):
        cls.entries: List[Entry] = []
        cls.dictionary: Dict[str, int] = dict()
        cls.matrix: List[List[float]] = []
        nltk.download('wordnet')
        nltk.download('stopwords')

    def fit(self, _load_dumped=False):
        loaded = False
        if _load_dumped:
            try:
                self.entries = self.load_dumped(const.DUMP_FP_ENTRIES)
                self.dictionary = self.load_dumped(const.DUMP_FP_DICTIONARY)
                self.matrix = self.load_dumped(const.DUMP_FP_MATRIX)
                print('Loaded dumped files')
                loaded = True
            except FileNotFoundError:
                pass
        if not loaded:
            self.init_entries()
            self.preprocess_entries()
            self.create_matrix(self.create_IDFs())

    @abstractmethod
    def init_entries(self):
        raise NotImplementedError

    def preprocess_entries(self):
        stopwords_en = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        words = set()

        for entry in self.entries:
            entry.additional_info['words_dict'] = defaultdict(int)
            entry.additional_info['words_cnt'] = 0
            words |= self.preprocess(entry, lemmatizer, stopwords_en)

        for wi, word in enumerate(words):
            self.dictionary[word] = wi

        self.dump(const.DUMP_FP_ENTRIES, self.entries)
        self.dump(const.DUMP_FP_DICTIONARY, self.dictionary)

    def preprocess(self, entry, lemmatizer, stopwords_en):
        text = entry.text
        text = text.lower() # małe litery
        text = re.sub(r'["`\'()\[\]{}|,.:;!?\-_–]+', ' ', text) # znaki nieniosące treści
        text = re.sub(r'\s\s+', ' ', text) # usunięcie zbędnych białych znaków
        for word in text.split():
            _word = lemmatizer.lemmatize(word, pos='v')
            if _word == word: _word = lemmatizer.lemmatize(word, pos='n')
            if _word == word: _word = lemmatizer.lemmatize(word, pos='a')
            if _word not in stopwords_en:
                entry.additional_info['words_dict'][_word] += 1
                entry.additional_info['words_cnt'] += 1

        return entry.additional_info['words_dict'].keys()

    def create_IDFs(self):
        entries_containing_word = defaultdict(int)
        for entry in self.entries:
            for word, occ_cnt in entry.additional_info['words_dict'].items():
                if occ_cnt > 0:
                    entries_containing_word[word] += 1

        N = len(self.dictionary)
        IDFs = np.array([0]*N, dtype=float)
        for word, wi in self.dictionary.items():
            IDFs[wi] = np.log(N/entries_containing_word[word])
        del entries_containing_word

        return IDFs

    def create_matrix(self, IDFs):
        for entry in self.entries:
            self.matrix.append(np.array([0]*len(self.dictionary), dtype=float))
            for word, occ_cnt in entry.additional_info['words_dict'].items():
                self.matrix[-1][self.dictionary[word]] = occ_cnt / \
                    entry.additional_info['words_cnt'] * \
                    IDFs[self.dictionary[word]]

            self.matrix[-1] = csr_matrix(self.matrix[-1])
            del entry.additional_info['words_dict']
            del entry.additional_info['words_cnt']
            
        del IDFs
        self.dump(const.DUMP_FP_MATRIX, self.matrix)

    def dump(self, dump_fp, obj):
        with open(dump_fp, 'wb+') as dump_f:
            pickle.dump(obj, dump_f)

    def load_dumped(self, dump_fp):
        with open(dump_fp, 'rb') as dump_f:
            obj = pickle.load(dump_f)
        return obj
