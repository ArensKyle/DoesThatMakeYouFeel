import re

def massage(data, col):

    tweets = []
    for lines in data:
        tweets.append(lines.split("\t"))
        tempTweet = tweets[-1][col]
        tempTweet = re.sub("(http|https):\/\/t\..*?\s", "httpstco", tempTweet)
        #tempTweet = re.sub("([a-zA-Z])\1{2,}")
