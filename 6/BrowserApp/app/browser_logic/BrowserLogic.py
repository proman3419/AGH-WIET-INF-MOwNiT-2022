from app.browser_logic.Entry import Entry
import app.browser_logic.const as const
from abc import ABC, abstractmethod
from typing import List, Dict
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import pickle
from collections import defaultdict
import scipy as sci
import numpy as np
from pathlib import Path
import os


class BrowserLogic(ABC):
    def __init_subclass__(cls, **kwargs):
        cls.entries: List[Entry] = []
        cls.dictionary: Dict[str, int] = dict()
        cls.matrix: List[List[float]] = []
        nltk.download('wordnet')
        nltk.download('stopwords')

    def fit(self, _load_dumped=False):
        Path(os.path.join('tmp', self.name)).mkdir(parents=True, exist_ok=True)
        loaded = False
        if _load_dumped:
            try:
                self.entries = self.load_dumped(const.DUMP_FB_ENTRIES)
                self.dictionary = self.load_dumped(const.DUMP_FB_DICTIONARY)
                self.matrix = self.load_dumped(const.DUMP_FB_MATRIX)
                print('Loaded cached files')
                loaded = True
            except FileNotFoundError as e:
                pass
        if not loaded:
            self.init_entries()
            self.preprocess_entries()
            self.create_matrix(self.create_IDFs())
        print('Finished fitting')

    def search(self, keywords, results_cnt, print_text=True):
        keywords = set(keywords)
        scores_entry_ids = self.search_raw(keywords, results_cnt)
        print(f'Results for: {keywords}')
        i = 0
        for score, entry in scores_entry_ids:
            print(f'>>> Result {i}, scored {score}')
            entry.pretty_print(print_text)
            print()
            i += 1

    def search_raw(self, keywords, results_cnt, noise_reduction=False,
                   noise_reduction_value=0.2):
        keywords = set(keywords)
        query_vec = np.zeros(len(self.dictionary), dtype=float)
        for kw in keywords:
            if kw in self.dictionary:
                query_vec[self.dictionary[kw]] = 1
        query_norm = np.linalg.norm(query_vec)
        matrix = self.matrix
        N = len(self.entries)

        if noise_reduction:
            # normalizacja
            query_vec /= query_norm
            for ei in range(N):
                matrix[ei] /= np.linalg.norm(matrix[ei].A)
            k = int(noise_reduction_value*N)
            U, D, V_T = sci.sparse.linalg.svds(matrix, k=k)
            matrix = sci.sparse.csr_matrix(U @ np.diag(D) @ V_T)

        query_vec_T = query_vec.T
        scores = [None]*N
        for ei in range(N):
            scores[ei] = (self.calculate_score(matrix, query_vec_T, 
                                               query_norm, ei), self.entries[ei])

        return sorted(scores)[N-1:-results_cnt-1:-1]

    def calculate_score(self, matrix, query_vec_T, query_norm, entry_id):
        entry_vec = matrix[entry_id].A[0]
        numerator = query_vec_T @ entry_vec
        denominator = query_norm * np.linalg.norm(entry_vec)
        return numerator / denominator

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

        self.dump(const.DUMP_FB_DICTIONARY, self.dictionary)

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
        IDFs = np.zeros(N, dtype=float)
        for word, wi in self.dictionary.items():
            IDFs[wi] = np.log(N/entries_containing_word[word])
        del entries_containing_word

        return IDFs

    def create_matrix(self, IDFs):
        for entry in self.entries:
            self.matrix.append(np.zeros(len(self.dictionary), dtype=float))
            for word, occ_cnt in entry.additional_info['words_dict'].items():
                self.matrix[-1][self.dictionary[word]] = occ_cnt / \
                    entry.additional_info['words_cnt'] * \
                    IDFs[self.dictionary[word]]

            self.matrix[-1] = sci.sparse.csr_matrix(self.matrix[-1])
            del entry.additional_info['words_dict']
            del entry.additional_info['words_cnt']

        del IDFs
        self.matrix = sci.sparse.vstack(self.matrix)
        self.dump(const.DUMP_FB_ENTRIES, self.entries)
        self.dump(const.DUMP_FB_MATRIX, self.matrix)

    def get_fp(self, fb):
        return os.path.join('tmp', self.name, f'{fb}.pkl')

    def dump(self, dump_fb, obj):
        with open(self.get_fp(dump_fb), 'wb+') as dump_f:
            pickle.dump(obj, dump_f)

    def load_dumped(self, dump_fb):
        with open(self.get_fp(dump_fb), 'rb') as dump_f:
            obj = pickle.load(dump_f)
        return obj
