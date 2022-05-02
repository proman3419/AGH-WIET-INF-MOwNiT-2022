from Entry import Entry
from BrowserLogic import BrowserLogic
import datasets
from collections import defaultdict


class WikipediaBrowserLogic(BrowserLogic):
    def init_entries(self):
        raw_entries = datasets.load_dataset("wikipedia", "20220301.simple", 
                                            split='train[:5%]')
        print(len(raw_entries))
        for raw_entry in raw_entries:
            self.entries.append(Entry(raw_entry['id'], raw_entry['url'],
                                      raw_entry['title'], raw_entry['text'], 
                                      words_dict=defaultdict(int), words_vec=[],
                                      additional_info=None))