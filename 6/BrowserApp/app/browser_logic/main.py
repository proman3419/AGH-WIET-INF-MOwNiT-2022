from WikipediaBrowserLogic import WikipediaBrowserLogic
import numpy as np


WBL = WikipediaBrowserLogic()
WBL.fit(_load_dumped=True)
WBL.search(['war'], 5, print_text=False)
