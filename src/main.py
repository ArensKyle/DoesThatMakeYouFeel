import massage, sys

def main():
    # Load the data files and massage
    datafileA = open('../data/2016downloaded4-subtask A.tsv')
    taskARet = massage.massage(datafileA, 2)
    taskA = taskARet[0]
    taskABoW = taskARet[1]
    for tweet in taskA:
        for token in tweet.tokens:
            print(tweet.id, token.attrs, token.word)
    # rip out relevant part for tokenizer


if __name__ == '__main__':
    sys.exit(main())