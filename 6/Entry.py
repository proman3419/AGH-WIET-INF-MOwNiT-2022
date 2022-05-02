from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Entry:
    id: int
    url: str
    title: str
    text: str
    words_dict: Dict[str, int]
    words_vec: List[int]
    additional_info: dict
