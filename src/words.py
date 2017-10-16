
_FEAT_PTR = 0
def new_feat():
    global _FEAT_PTR
    feature = _FEAT_PTR
    _FEAT_PTR = _FEAT_PTR + 1
    return feature

def feat_len():
    global _FEAT_PTR
    return _FEAT_PTR

class Word:
    def __init__(self, word, pos):
        self.word = word
        self.pos = pos
        self.attrs = [0 for x in range(feat_len())]
