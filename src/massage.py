import re, nltk, words, tweet

SPELL_F = words.new_feat()
QUESTION_F = words.new_feat()
EXCLAMATION_F = words.new_feat()
HASHTAG_F = words.new_feat()
PERIOD_F = words.new_feat()




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
            twit_id = lines.split("\t")[0]
            subject = ""
        else:
            sentiment = lines.split("\t")[2]
            twit_id = lines.split("\t")[0]
            subject = lines.split("\t")[1]

        w_tokens = []
        for w in tokens:
            w_tokens.append(word(w))

        tweet_obj = tweet(w_tokens, twit_id, body, sentiment, subject)

        tweets.append(tweet_obj)

    return tweets
