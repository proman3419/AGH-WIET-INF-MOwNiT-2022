import const
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Entry:
    url: str
    title: str
    text: str
    additional_info: dict

    def pretty_print(self, print_text=True):
        print(f'url: {self.url}')
        print(f'title: {self.title}')
        if print_text:
            text_len = min(len(self.text), const.ENTRY_TEXT_PRINT_LEN)
            print(f'text: {self.text[:text_len]}...')
        print(f'url: {self.url}')
