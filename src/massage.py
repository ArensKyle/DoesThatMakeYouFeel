from nltk.tokenize import TweetTokenizer
from tokenParse import annotateTokens
import re, nltk, words, tweet

SIGNIFICANT = 3

def massage(data, col):
    bag_of_words = {}
    sig_words = {}
    tweets = []
    x=0
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

        w_tokens = []
        w_punc = []
        pos_tags = nltk.pos_tag(tokens)

        #iterate through all tokens in the body of a given tweet
        for idx in range(len(tokens)):
            start_punc = -1
            pos = pos_tags[idx][1]

            # create the Word
            w_tokens.append(words.Word(
                tokens[idx], # word
                pos) # pos
            )

            # update bag of words
            key = pos_tags[idx][0] + "_" + (pos)
            bag_of_words[key] = bag_of_words.get(key, 0) + 1


        #turn bag of words into index of significant words for NN
        #word index 0 = padding
        #word index 1 = never seen
        word_index = 2
        for word in bag_of_words:
            if(bag_of_words.get(word) > SIGNIFICANT):
                sig_words[word] = word_index
                word_index += 1

        #calls the token parsing functionality in order to determine features
        #vector for each word
        annotateTokens(w_tokens)
        tweet_obj = tweet.Tweet(w_tokens, twit_id, body, sentiment, subject)

        #add the analyzed tweet into the final array for the neural net
        tweets.append(tweet_obj)

    return tweets, sig_words
