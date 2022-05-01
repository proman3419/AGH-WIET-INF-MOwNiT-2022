from dataclasses import dataclass
from typing import List


@dataclass
class Entry:
    id: int
    url: str
    title: str
    text: str
    words_set: set
    words_vec: List[int]
    additional_info: dict
