from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Entry:
    id: int
    url: str
    title: str
    text: str
    additional_info: dict
