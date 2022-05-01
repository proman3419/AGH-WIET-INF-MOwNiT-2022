from Entry import Entry
from BrowserLogic import BrowserLogic
from datasets import load_dataset


class WikipediaBrowserLogic(BrowserLogic):
    def load_entries(self):
        raw_entries = load_dataset("wikipedia", "20220301.simple")['train']
        for raw_entry in raw_entries:
            self.entries.append(Entry(raw_entry['id'], raw_entry['url'],
                                      raw_entry['title'], raw_entry['text'], 
                                      words_set=set(), words_vec=[],
                                      additional_info=None))
