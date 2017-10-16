from nltk.tokenize import TweetTokenizer
import re, nltk, words, tweet

def massage(data, col):
    bagOfWords = {}
    tweets = []
    x=0
    for lines in data:
        tknzr = TweetTokenizer()

        body = lines.split("\t")[-1]
        body = re.sub("(http|https):\/\/t\..*?\s", "httpstco", body)
        #body = re.sub("([a-zA-Z])\1{2,}", body)
        tokens = tknzr.tokenize(body)

        if (len(lines.split("\t")) == 3):
            sentiment = lines.split("\t")[1]
            twit_id = lines.split("\t")[0]
            subject = ""
        else:
            sentiment = lines.split("\t")[2]
            twit_id = lines.split("\t")[0]
            subject = lines.split("\t")[1]

        w_tokens = []
        pos_tags = nltk.pos_tag(tokens)
        
        for idx in range(len(tokens)):
            # create the Word
            w_tokens.append(words.Word(
                tokens[idx], # word
                pos_tags[idx][1]) # pos
            )

            # update bag of words
            key = pos_tags[idx][0] + (pos_tags[idx][1])
            bagOfWords[key] = bagOfWords.get(key, 0) + 1

        #for w in tokens:
        #    w_tokens.append(words.Word(w))

        tweet_obj = tweet.Tweet(w_tokens, twit_id, body, sentiment, subject)

        tweets.append(tweet_obj)

    return tweets, bagOfWords
