from app.browser_logic.Entry import Entry
from app.browser_logic.BrowserLogic import BrowserLogic
import datasets


class WikipediaBrowserLogic(BrowserLogic):
    def __init__(self):
        self.name = 'wikipedia'
    
    def init_entries(self):
        raw_entries = datasets.load_dataset("wikipedia", "20220301.simple", 
                                            split='train[:2000]')
        for raw_entry in raw_entries:
            self.entries.append(Entry(raw_entry['url'], raw_entry['title'], 
                                      raw_entry['text'], additional_info={}))
