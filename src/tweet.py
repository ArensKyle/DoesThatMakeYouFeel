
class Tweet:
  def __init__(self, tokens=[], twitterId='', literal='', sentiment='', features={}):
    self.tokens = tokens
    self.id = twitterId
    self.literal = literal
    self.sentiment = sentiment
    self.attrs = features

