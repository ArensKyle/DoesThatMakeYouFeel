import words_to_vectors as wtv
import conjunction as cnj
import massage, sys

def main():
    train()
    test()

def train():
    # Load the data files and massage
    datafile_A = open('../data/2016downloaded4-subtask A.tsv')
    print("entering massage")
    task_A_ret = massage.massage(datafile_A, 2)
    print("leaving massage")
    task_A = task_A_ret[0]
    task_A_word_index = task_A_ret[1]

    conj_returns = conj_classify_tweets(task_A)
    task_A = conj_returns[0]
    conj_A = conj_returns[1]

    #for tweet in task_A:
    #    for token in tweet.tokens:
    #        print(tweet.id, token.attrs, token.word)
    # rip out relevant part for tokenizer

    print("entering word to vec")
    vector_returns_A = wtv.sig_vec(task_A, task_A_word_index)
    print("leaving word to vec")
    word_map_A = vector_returns_A[0]
    feat_map_A = vector_returns_A[1]

    print("entering word to vec")
    conj_vector_returns_A = wtv.sig_vec(conj_A, task_A_word_index)
    print("leaving word to vec")
    conj_word_map_A = vector_returns_A[0]
    conj_feat_map_A = vector_returns_A[1]


def test():



if __name__ == '__main__':
    sys.exit(main())
