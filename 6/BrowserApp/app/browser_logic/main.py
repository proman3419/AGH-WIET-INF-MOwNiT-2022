from WikipediaBrowserLogic import WikipediaBrowserLogic
import numpy as np


WBL = WikipediaBrowserLogic()
WBL.fit(_load_dumped=True)
# WBL.search(['war'], 5, print_text=False)
WBL.search_raw(['water'], 10, noise_reduction=True, noise_reduction_value=0.3)
