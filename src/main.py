import words_to_vectors as wtv
import chunker as c
import massage, sys
import words as w
import netutil
import math

import sentimentnet

import tensorflow as tf

SIGNIFICANT = 3

subtask = "a"

categoryCount = {
    "a": 3,
    "b": 2,
    "c": 5
}

trainData = {
    "a": "../data/Subtask_A/twitter-2016dev-A.txt.download",
    "b": "../data/Subtasks_BD/twitter-2016dev-BD.txt.download",
    "c": "../data/Subtasks_CE/twitter-2016dev-CE.txt.download"
}

testData = {
    "a": "../data/Subtask_A/twitter-2016devtest-A.txt.download",
    "b": "../data/Subtasks_BD/twitter-2016devtest-BD.txt.download",
    "c": "../data/Subtasks_CE/twitter-2016devtest-CE.txt.download"
}

def main():
    m = Model(subtask.upper(), categoryCount[subtask])
    with tf.Session() as sess:
        #print("Beginning training")
        m.train(sess, [trainData[subtask]], 10, 10)
        #print("Beginning testing")
        m.test(sess, [testData[subtask]])

class Model:
    def __init__(self, task, categories):
        self.task = task
        self.categories = categories

        self.word_index_map = {}
        self.bag_of_words = {}

        self.sentinet = None

    def loadTweets(self, filenames, bag_it=False, tweets=[]):
        if type(filenames) is list:
            for f in filenames:
                self.loadTweets(f, bag_it, tweets)
        else:
            with open(filenames) as f:
                if bag_it:
                    return massage.massage(f, tweets=tweets,
                            bag_of_words=self.bag_of_words)
                else:
                    return massage.massage(f, tweets=tweets,
                            bag_of_words=None)

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

    def train(self, sess, files, batchsize, batchrounds):
        tweets = self.loadTweets(files, True)
        self.buildWordIndexMap()

        print("Word Vocab Size: {}".format(len(self.word_index_map) + 2))


        batchrounds = math.ceil(len(tweets) * batchrounds / batchsize)
        tchunker = c.Chunker(tweets)
        
        self.sentinet = sentimentnet.create_graph(self.categories, len(self.word_index_map) + 2, w.feat_len())
        
        tw_input, tf_input, graph, variables = self.sentinet

        expected_input, optimize_graph = netutil.optimize(graph, self.categories)
        trainer = tf.train.AdamOptimizer(1e-2)
        
        trainfn = trainer.minimize(optimize_graph)
        # initialize nets
        sess.run(tf.variables_initializer(variables))
        sess.run(tf.global_variables_initializer())
       
        for i in range(batchrounds):
            tweetbatch = tchunker.batch(batchsize)
            words, feats, expected = wtv.sig_vec(tweetbatch, self.word_index_map, self.task)

            _, loss_val = sess.run([trainfn, optimize_graph], feed_dict={tw_input: words, tf_input: feats, expected_input: expected})
            if i % 10 == 0:
                print("LOSS: {} @ {}".format(loss_val, i))

    def test(self, sess, files):
        tweets = self.loadTweets(files, False)

        correct_results = 0

        tw_input, tf_input, graph, variables = self.sentinet

        y_ = tf.placeholder(tf.int32, [None, self.categories])
        validation_graph = tf.equal(tf.argmax(graph, 1), tf.argmax(y_, 1))
        prediction = tf.argmax(graph, 1)

        accuracy = tf.reduce_mean(tf.cast(validation_graph, tf.float32))
        w_vals, f_vals, expected = wtv.sig_vec(tweets, self.word_index_map, self.task)
    
        prediction_v, accuracy_v = sess.run([prediction, accuracy], feed_dict={tw_input: w_vals, tf_input: f_vals, y_: expected})
        print(accuracy_v)

if __name__ == '__main__':
    for x in range(1):
        main()
