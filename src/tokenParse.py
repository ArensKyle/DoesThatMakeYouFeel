import words

import enchant

SPELL_F = words.new_feat()
HASHTAG_F = words.new_feat()

PERIOD_F = words.new_feat()
QUESTION_F = words.new_feat()
EXCLAMATION_F = words.new_feat()
PERIODPERIOD_F = words.new_feat()
ELLIPSIS_F = words.new_feat()

d = enchant.Dict("en_US")

PUNCTUATION_SEPERATORS = ['.','..','...','!','?']
PUNCTUATION_MAPPING = {
  ".": PERIOD_F,
  "..": PERIODPERIOD_F,
  "...": ELLIPSIS_F,
  "!": EXCLAMATION_F,
  "?": QUESTION_F
}

#top level token parsing function. Calls more pecific functions
def annotateTokens(tokens):
  lastPuncGroupIdx = 0
  for idx in range(len(tokens)):
    annotateHashtag(tokens[idx])
    annotateSpelling(tokens[idx])
    annotatePunctuation(idx, lastPuncGroupIdx, tokens)

    #allows for the grouping of punctuation to account for repeating punctuation
    if (tokens[idx].word not in PUNCTUATION_SEPERATORS and idx > 0 and tokens[idx - 1].word in PUNCTUATION_SEPERATORS):
      # print("updating lastPuncGroup")
      lastPuncGroupIdx = idx

#parses and accounts for punctuation features
def annotatePunctuation(currentIdx, lastGroupIdx, tokens):
  if (tokens[currentIdx].word in PUNCTUATION_SEPERATORS):
    attr_idx = PUNCTUATION_MAPPING[tokens[currentIdx].word]
    for idx in range(lastGroupIdx, currentIdx):
      tokens[idx].attrs[attr_idx] += 1

#weight hashtags
def annotateHashtag(token):
  '''Take a token, and check to see if a token is a hashtag'''
  token.attrs[HASHTAG_F] = int(token.word[0] == "#" and len(token.word) > 1)

#correct or incorrect spelling as a feature
def annotateSpelling(token):
  '''Take a token, and check if is spelled correctly'''
  token.attrs[SPELL_F] = int(d.check(token.word))
  
