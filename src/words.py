
#Will generate a uniue index for each feature used for analysis
_FEAT_PTR = 0
def new_feat():
    global _FEAT_PTR
    feature = _FEAT_PTR
    _FEAT_PTR = _FEAT_PTR + 1
    return feature

#returns the number of features that are being used
def feat_len():
    global _FEAT_PTR
    return _FEAT_PTR

#class to represent one word or token.
#each word has a weighted list of relevant attributes
#part of speech identifies words that are spelled same, but are different words
# like the verb run, and the noun run
class Word:
    def __init__(self, word, pos):
        self.word = word
        self.pos = pos
        self.attrs = [0 for x in range(feat_len())]
