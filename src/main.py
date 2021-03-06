import words_to_vectors as wtv
import chunker as c
import massage, sys
import words as w
import netutil
import math
import argparse

import sentimentnet

import tensorflow as tf

SIGNIFICANT = 3
TRAIN_RATE = 1e-1
KEEP_PROB = 1
ROUNDS = 1

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

competeData = {
    "a": "../data/Subtask_A/twitter-2016test-A.txt.download",
    "b": "../data/Subtasks_BD/twitter-2016test-BD.txt.download",
    "c": "../data/Subtasks_CE/twitter-2016test-CE.txt.download"
}

categoryMapping = {
    "A": {
        2: "positive",
        1: "neutral",
        0: "negative",
    },
    "B": {
        1: "positive",
        0: "negative",
    },
    "C": {
        4: "2",
        3: "1",
        2: "0",
        1: "-1",
        0: "-2",
    }
}

def main():
    p = argparse.ArgumentParser(description='Does some math')
    p.add_argument('subtask', choices=['a', 'b', 'c'])
    a = p.parse_args()
    m = Model(a.subtask.upper(), categoryCount[a.subtask])
    with tf.Session() as sess:
        #print("Beginning training")
        m.train(sess, trainData[a.subtask], 10, ROUNDS)
        #print("Beginning testing")
        print("calling test with ", competeData[a.subtask])
        m.test(sess, competeData[a.subtask])

class Model:
    def __init__(self, task, categories):
        self.task = task
        self.categories = categories

        self.word_index_map = {}
        self.bag_of_words = {}

        self.sentinet = None

    def loadTweets(self, filename, bag_it=False):
        with open(filename) as f:
            return massage.massage(f, bag_of_words=(self.bag_of_words if bag_it else None))

    def buildWordIndexMap(self):
        self.word_index_map = {}

        #turn bag of words into index of significant words for NN
        #word index 0 = padding
        #word index 1 = never seen
        #word index 2 = reserved for subject
        word_index = 3
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
        
        self.sentinet = sentimentnet.create_graph(self.categories, len(self.word_index_map) + 3, w.feat_len())
        
        tw_input, tf_input, graph, keep_prob, variables = self.sentinet
        

        expected_input, optimize_graph = netutil.optimize(graph, self.categories)
        validation_graph = tf.equal(tf.argmax(graph, 1), tf.argmax(expected_input, 1))

        accuracy = tf.reduce_mean(tf.cast(validation_graph, tf.float32))

        trainer = tf.train.AdamOptimizer(TRAIN_RATE)
        
        trainfn = trainer.minimize(optimize_graph)
        # initialize nets
        sess.run(tf.variables_initializer(variables))
        sess.run(tf.global_variables_initializer())
       
        for i in range(batchrounds):
            tweetbatch = tchunker.batch(batchsize)
            words, feats, expected = wtv.sig_vec(tweetbatch, self.word_index_map, self.task)

            sess.run(trainfn, feed_dict={tw_input: words, tf_input: feats, expected_input: expected, keep_prob: KEEP_PROB})
            if i % 10 == 0:
                accuracy_f = sess.run(accuracy, feed_dict={tw_input: words, tf_input: feats, expected_input: expected, keep_prob: 1})
                print("accuracy: {:.1f} @ {}".format(accuracy_f, i))

    def test(self, sess, files):
        tweets = self.loadTweets(files, False)
        correct_results = 0
        print("test categories", self.categories)
        tw_input, tf_input, graph, keep_prob, variables = self.sentinet

        y_ = tf.placeholder(tf.int32, [None, self.categories])
        validation_graph = tf.equal(tf.argmax(graph, 1), tf.argmax(y_, 1))
        prediction = tf.argmax(graph, 1)

        accuracy = tf.reduce_mean(tf.cast(validation_graph, tf.float32))
        w_vals, f_vals, expected = wtv.sig_vec(tweets, self.word_index_map, self.task)
    
        prediction_v, accuracy_v = sess.run([prediction, accuracy], feed_dict={tw_input: w_vals, tf_input: f_vals, y_: expected, keep_prob: 1})
        print(accuracy_v)
        with open('results.txt', 'w') as rf:
            for idx in range(len(prediction_v)):
                if self.task == "A":
                    rf.write(tweets[idx].id + "\t" + categoryMapping[self.task][prediction_v[idx]] + "\n")
                else:
                    rf.write(tweets[idx].id + "\t" + tweets[idx].subject + "\t" + categoryMapping[self.task][prediction_v[idx]] + "\n")
        
        with open('master.txt', 'w') as rf:
            for idx in range(len(prediction_v)):
                if self.task == "A":
                    rf.write(tweets[idx].id + "\t" + tweets[idx].sentiment + "\n")
                else:
                    rf.write(tweets[idx].id + "\t" + tweets[idx].subject + "\t" + tweets[idx].sentiment + "\n")
        
        

if __name__ == '__main__':
    for x in range(1):
        main()
