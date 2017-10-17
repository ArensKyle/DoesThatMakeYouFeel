import words_to_vectors as wtv
import conjunction as cnj
import massage, sys

import tensorflow as tf

def main():
    m = Model()

    m.train(["trainfile"])
    m.test(["testfile"])

class Model:
    def __init__(self):
        self.word_index_map = {}
        self.bag_of_words = {}

        self.sentinet = None
        self.conjnet = None

    def loadTweets(self, filenames, bag_it=False, tweets=[]):
        if type(filenames) is list:
            for f in filenames:
                self.loadTweets(f, bag_it, tweets)
        else:
            with open(filename) as f:
                if bag_it:
                    bow = None
                else:
                    bow = self.bag_of_words
                return massage.massage(f, tweets=tweets,
                        bag_of_words=bow)

        return tweets, sig_words

    def buildWordIndexMap(self):
        self.word_index_map = {}

        #turn bag of words into index of significant words for NN
        #word index 0 = padding
        #word index 1 = never seen
        word_index = 2
        for word in self.bag_of_words:
            if(self.bag_of_words[word] > SIGNIFICANT):
                self.word_index_map[word] = word_index
                word_index += 1

    def train(self, files):
        tweets, sig_words = self.loadTweets(files, True)
        self.buildW2VMap()

        tweets, conjunctives = conj_classify_tweets(tweets)

        # first, go through and train the main net

        # DO TRAINING!!!!
        # results will be stored in sentinet

        # DO MORE TRAINING!!!!
        # results will be stored in conjnet

    def test(self, files):
        tweets, sig_words = self.loadTweets(files, False)

        tweets, conjunctives = conj_classify_tweets(tweets)

        # run tweets through batch

        # flatten and run conjunctives through

if __name__ == '__main__':
    sys.exit(main())
