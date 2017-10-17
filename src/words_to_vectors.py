import numpy as np
import re, nltk, words, tweet

def sig_vec(tweets, sig_words, task):
    word_map = np.zeros([len(tweets), 70])
    feat_map = np.zeros([len(tweets), 70, words.feat_len()])
    if (task == "A"):
        categories = 3
    elif (task == "B" or task == "D"):
        categories = 2
    else:
        categories = 5
    expected_map = np.zeros([len(tweets), categories])


    tweet_index = 0
    for record in tweets:
        word_index = 0
        #for creating expected sentiment vector
        if (task == "A"):
            if (record.sentiment == "positive"):
                expected_index = 2
            elif (record.sentiment == "neutral"):
                expected_index = 1
            else:
                expected_index = 0
            expected_map[tweet_index, expected_index] = 1
        elif (task == "B" or task == "D"):
            if (record.sentiment == "positive"):
                expected_index = 1
            else:
                expected_index = 0
            expected_map[tweet_index, expected_index] = 1
        else:
            if (record.sentiment == 2):
                expected_index = 4
            elif (record.sentiment == 1):
                expected_index = 3
            elif (record.sentiment == 0):
                expected_index = 2
            elif (record.sentiment == -1):
                expected_index = 1
            else:
                expected_index = 0
            expected_map[tweet_index, expected_index] = 1
        for token in record.tokens:
            if (sig_words.get(token.word + "_" + token.pos)):
                word_map[tweet_index, word_index] = sig_words[token.word + "_" + token.pos]
            else:
                word_map[tweet_index, word_index] = 1

            #create 3rd dimensional vectors to hold feature values
            feature_index = 0
            for feat in token.attrs:
                feat_map[tweet_index, word_index, feature_index] = feat
                feature_index = feature_index + 1
            word_index = word_index + 1
        tweet_index = tweet_index + 1

    return word_map, feat_map, expected_map
