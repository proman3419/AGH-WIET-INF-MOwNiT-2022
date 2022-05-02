from WikipediaBrowserLogic import WikipediaBrowserLogic
import numpy as np


WBL = WikipediaBrowserLogic()
WBL.fit(_load_dumped=True)
print(WBL.entries[1].text)
print(WBL.matrix[1].toarray()[0][WBL.dictionary['month']])
print(WBL.matrix[1].toarray()[0][WBL.dictionary['august']])
