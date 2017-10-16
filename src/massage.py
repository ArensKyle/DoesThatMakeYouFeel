import re, nltk,

def massage(data, col):

    tweets = []
    for lines in data:
        tknzr = TweetTokenizer()

        body = lines.split("\t")[-1]
        body = re.sub("(http|https):\/\/t\..*?\s", "httpstco", body)
        #body = re.sub("([a-zA-Z])\1{2,}", body)
        tokens = tknzr.tokenize(body)

        if (len(lines.split("\t")) == 3):
            sentiment = lines.split("\t")[1]
            twitID = lines.split("\t")[0]
            subject = ""
        else:
            sentiment = lines.split("\t")[2]
            twitID = lines.split("\t")[0]
            subject = lines.split("\t")[1]

        tweetObj = tweet(tokens, twitID, body, sentiment, subject)

        tweets.append(tweetObj)

    return tweets
