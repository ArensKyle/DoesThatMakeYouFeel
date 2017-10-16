import words

import enchant

SPELL_F = words.new_feat()
QUESTION_F = words.new_feat()
EXCLAMATION_F = words.new_feat()
HASHTAG_F = words.new_feat()
PERIOD_F = words.new_feat()

d = enchant.Dict("en_US")


PUNCTUATION_SEPERATORS = ['.','..','...','!','?']

def annotateTokens(tokens):
  start_punc = tokens[0]
  for token in tokens:
    #if token in PUNCTUATION_SEPERATORS:

    token=token
  return tokens

def annotateHashtag(token):
  '''Take a token, and check to see if a token is a hashtag'''
  token.attrs[HASHTAG_F] = int(token.word[0] == "#" and len(token.word) > 1)

def annotateSpelling(token):
  '''Take a token, and check if is spelled correctly'''
  token.attrs[SPELL_F] = int(d.check(token))

def testAnnotate():
  token = words.Word("#test")
  annotateHashtag(token)
  print(HASHTAG_F)
  print(token.attrs)