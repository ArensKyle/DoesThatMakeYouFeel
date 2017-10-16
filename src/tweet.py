
class Tweet:
  def __init__(self, tokens=[], twitterId='', literal='', sentiment='', subject='', features={}):
    self.tokens = tokens
    self.id = twitterId
    self.literal = literal
    self.sentiment = sentiment
    self.attrs = features
    self.subject = subject

