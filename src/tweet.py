#class object used for holding one tweet
class Tweet:
  def __init__(self, tokens=[], twitterId='', literal='', sentiment='', subject=''):
    self.tokens = tokens #A list of all tokens present in the text body of the tweet
    self.id = twitterId #the unique ID associated with a tweet
    self.literal = literal #the text body of the tweet
    self.sentiment = sentiment #the given sentiment for training data, or the discerned sentiment for testing data
    self.subject = subject #the subject that is given with the tweet
