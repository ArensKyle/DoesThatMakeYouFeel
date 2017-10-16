
# this probably needs to be dynamically created
CONJUNCTIONS = ["for", "and", "nor", "but", "or", "yet", "so", "while", "although"]

class Conjunctive:
    """
    %param parent parent - can be either another conjunctive or a tweet
    %param l left side - list of tokens
    %param conj conjucate token between l and r
    %param r right side - list of tokens
    """
    def __init__(self, parent, l, conj, r):
        self.parent = parent
        self.l = l
        self.conj = conj
        self.r = r

def join_conjunctive(conjunctive, joinbatcher, evalbatcher):
    if type(conjunctive.l) is Conjunctive:
        lv = join_conjunctive(conjunctive.l, joinbatcher, evalbatcher)
    else:
        lv = evalbatcher(conjunctive.l)

    if type(conjunctive.r) is Conjunctive:
        rv = join_conjunctive(conjunctive.r, joinbatcher, evalbatcher)
    else:
        rv = evalbatcher(conjunctive.r)

    return joinbatcher(lv, conjunctive.conj, rv)


def split_tweet_to_conj_tree(tweet):
    # this should probably use a parsetree, but oh well!
    # we just instead go in reverse order
    last_conj = None
    pool = []
    for i in range(len(tweet) - 1, -1, -1):
        element = tweet.tokens[i]
        if element.pos == "CONJ":
            # split!
            if last_conj is not None:
                l = last_conj
                # we are unsure what our left is yet
                last_conj = Conjunctive(tweet, None, element, l)
                l.l = pool
                l.parent = last_conj
            else:
                last_conj = Conjunctive(tweet, None, element, pool)

            pool = [] # reset pool

    if last_conj is None:
        return tweet
    else:
        last_conj.l = pool
        return last_conj
