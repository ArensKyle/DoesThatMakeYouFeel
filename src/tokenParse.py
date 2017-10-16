import words

SPELL_F = words.new_feat()
QUESTION_F = words.new_feat()
EXCLAMATION_F = words.new_feat()
HASHTAG_F = words.new_feat()
PERIOD_F = words.new_feat()

def annotateTokens(tokens):
  for token in tokens:
    #do stuff
    token=token
  return tokens

def annotateHashtag(token):
  '''Take a token, and check to see if a token is a hashtag'''
  token.attrs[HASHTAG_F] = int(token.word[0] == "#" and len(token.word) > 1)

def testAnnotate():
  token = words.Word("#test")
  annotateHashtag(token)
  print(HASHTAG_F)
  print(token.attrs)