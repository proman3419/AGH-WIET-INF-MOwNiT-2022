from WikipediaBrowserLogic import WikipediaBrowserLogic


WBL = WikipediaBrowserLogic()
WBL.fit(_load_dumped=True)
print(WBL.entries[0])
print(len(WBL.dictionary))
# print(WBL.dictionary)
# print(len(WBL.dictionary))
