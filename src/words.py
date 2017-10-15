
_ATTR_PTR = 0
def new_attr():
    global _ATTR_PTR
    a = _ATTR_PTR
    _ATTR_PTR = _ATTR_PTR + 1
    return a

def attr_len():
    global _ATTR_PTR
    return _ATTR_PTR

class Word:
    def __init__(self, word):
        self.word = word
        self.attrs = [0 for x in range(attr_len)]

