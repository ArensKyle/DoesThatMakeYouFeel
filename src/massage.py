from nltk.tokenize import TweetTokenizer
from tokenParse import create_and_annotate_words
import re, nltk, words, tweet

SIGNIFICANT = 3

def massage(data, bag_of_words={}):
    tweets = []
    tknzr = TweetTokenizer()
    for lines in data:

        #split record into tab separated values
        body = lines.split("\t")[-1]

        if (body == "Not Available\n"):
            continue

        body = re.sub("(http|https):\/\/t\..*?\s", "httpstco", body)
        #body = re.sub("([a-zA-Z])\1{2,}", body)
        tokens = tknzr.tokenize(body)

        #Conditional for the differnt formatting between task A data and data
        #for other 4 tasks
        if (len(lines.split("\t")) == 3):
            sentiment = lines.split("\t")[1]
            twit_id = lines.split("\t")[0]
            subject = ""
        else:
            sentiment = lines.split("\t")[2]
            twit_id = lines.split("\t")[0]
            subject = lines.split("\t")[1]

        # create feature vectors
        w_tokens = create_and_annotate_words(tokens)
        # print('printing w_tokens: ', w_tokens)
        # update bag of words from tokens.
        if bag_of_words is not None:
            for token in w_tokens:
                key = token.word + "_" + token.pos
                bag_of_words[key] = bag_of_words.get(key, 0) + 1

        tweet_obj = tweet.Tweet(w_tokens, twit_id, body, sentiment, subject)

        #add the analyzed tweet into the final array for the neural net
        tweets.append(tweet_obj)

    return tweets
