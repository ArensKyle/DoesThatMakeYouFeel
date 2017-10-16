
class Tweet:
  def __init__(self, tokens=[], twitterId='', literal='', sentiment='', subject=''):
    self.tokens = tokens
    self.id = twitterId
    self.literal = literal
    self.sentiment = sentiment
    self.subject = subject
