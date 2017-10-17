import words_to_vectors as wtv
import chunker as c
import massage, sys


import tensorflow as tf

def main():
    m = Model("A", 3)

    m.train(["trainfile"])
    m.test(["testfile"])

class Model:
    def __init__(self, task, categories):
        self.task = task
        self.categories

        self.word_index_map = {}
        self.bag_of_words = {}

        self.sentinet = None

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

        return tweets

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

    def train(self, files, batchrounds, batchsize):
        tweets = self.loadTweets(files, True)
        self.buildWordIndexMap()

        tchunker = c.Chunker(tweets)

        # train main net
        with tf.Session() as sess:
            # initialize nets
            sess.run(tf.global_variables_initializer())
            self.sentinet = sentimentnet.create_graph(self.categories, len(self.word_index_map), words.feat_len())

            tw_input, tf_input, graph = self.sentinet

            expected_input, optimize_graph = netutil.optimize(graph, self.categories)
            trainfn = netutil.train(optimize_graph)
           
            for i in range(batchrounds):
                tweetbatch = tchunker.batch(batchsize)
                words, feats, expected = wtv.sig_vec(tweetbatch, self.word_index_map, task)

                sess.run(trainfn, feed_dict={tw_input: words, tf_input: feats, expected_input: expected})

    def test(self, files, batchrounds, batchsize):
        tweets = self.loadTweets(files, False)
        tchunker = c.Chunker(tweets)

        correct_results = 0

        with tf.Session() as sess:
            tw_input, tf_input, graph = self.sentinet

            y_ = tf.placeholder(tf.float32, [None, self.categories])
            validation_graph = netutil.correct_prediction(graph, y_)
            for i in range(batchrounds):
                tweetbatch = tchunker.batch(batchsize)
                w_vals, f_vals, expected = wtv.sig_vec(tweetbatch, sig_words, task)
            
                correct_results += sess.run(validation_graph, feed_dict={tw_input: w_vals, tf_input: f_vals, y_: expected})
        print("Accuracy: {}".format(correct_results / len(tweets)))

if __name__ == '__main__':
    sys.exit(main())
