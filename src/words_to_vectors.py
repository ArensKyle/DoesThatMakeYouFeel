import numpy as np
import re, nltk, words, tweet

def sig_vec(tweets, sig_words):
t    word_map = np.zeros([len(tweets), 70])
    feat_map = np.zeros([len(tweets), 70, words.feat_len()])
    print(word_map.size)
    print(feat_map.size)


    tweet_index = 0
    for record in tweets:
        word_index = 0
        for token in record.tokens:
            if (sig_words.get(token.word + "_" + token.pos)):
                word_map[tweet_index, word_index] = sig_words[token.word + "_" + token.pos]
            else:
                word_map[tweet_index, word_index] = 1

            feature_index = 0
            for feat in token.attrs:
                feat_map[tweet_index, word_index, feature_index] = feat
                feature_index = feature_index + 1
            word_index = word_index + 1
        tweet_index = tweet_index + 1

    return word_map, feat_map
