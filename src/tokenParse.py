# Brown PoS Corpus dict
# n - noun
# v - verb
# a - adjective
# s - adjective satellite
# r - adverb


import enchant, nltk, words

from nltk.corpus import sentiwordnet as swn

SPELL_F = words.new_feat()
HASHTAG_F = words.new_feat()

PERIOD_F = words.new_feat()
QUESTION_F = words.new_feat()
EXCLAMATION_F = words.new_feat()
PERIODPERIOD_F = words.new_feat()
ELLIPSIS_F = words.new_feat()

POSITIVE_F = words.new_feat()
NEGATIVE_F = words.new_feat()
OBJECTIVE_F = words.new_feat()

CCLEFT_F = words.new_feat()
CCRIGHT_F = words.new_feat()
CCSLEFT_F = words.new_feat()
CCSRIGHT_F = words.new_feat()
DTXLEFT_F = words.new_feat()
DTXRIGHT_F = words.new_feat()

BROWN_TO_SYNSET = {
  "NN": 'n',
  "NN$": 'n',
  "NNS": 'n',
  "NNS$": 'n',
  "VB": 'v',
  "VBD": 'v',
  "VBG": 'v',
  "VBN": 'v',
  "VBP": 'v',
  "VBZ": 'v',
  "JJ": 'a',
  "JJR": 'a',
  "JJS": 'a',
  "RB": 'r',
  "RBR": 'r',
  "RBS": 'r',
}

spell_dict = enchant.Dict("en_US")

PUNCTUATION_SEPERATORS = ['.','..','...','!','?']
PUNCTUATION_MAPPING = {
  ".": PERIOD_F,
  "..": PERIODPERIOD_F,
  "...": ELLIPSIS_F,
  "!": EXCLAMATION_F,
  "?": QUESTION_F
}

def create_and_annotate_words(tokens):
  # We already have the tokens, now create the word type.
  w_tokens = []

  #iterate through all tokens in the body of a given tweet
  for idx in range(len(tokens)):
      # create the Word
      token = words.Word(
        tokens[idx], # word
        '') # empty pos, will fill in later.

      # do annotating that can be done on just a token.
      annotateHashtag(token) # check if hashtag
      annotate_and_correct_spelling(token) # update spelling
      
      w_tokens.append(words.Word(
          tokens[idx], # word
          '') # empty pos
      )

  # now that we have "corrected" spelling, we can properly pos tag the token list.
  update_pos_tags(w_tokens)

  # now annotate punctuation.
  annotate_remaining_features(w_tokens)
  return w_tokens

def get_words_from_token_list(token_list):
  tokens = []
  for token in token_list:
    tokens.append(token.word)
  return list(tokens)
  
def update_pos_tags(token_list):
  # convert a list of tokens to just a list of words.
  word_list = get_words_from_token_list(token_list)
  pos_tags = nltk.pos_tag(word_list)
  for idx in range(len(token_list)):
    token_list[idx].pos = pos_tags[idx][1]

def annotate_remaining_features(token_list):
  ''' takes in a list of tokens and annotates the punctuation vector '''
  lastPuncGroupIdx = 0
  lastCCGroupIdx = 0
  for idx in range(len(token_list)):
    annotatePunctuation(idx, lastPuncGroupIdx, token_list)
    annotate_conjunction(idx, lastCCGroupIdx, token_list)
    annotate_sentiment(token_list[idx])
   
    # allows for the grouping of punctuation to account for repeating punctuation
    if (token_list[idx].word not in PUNCTUATION_SEPERATORS and idx > 0 and token_list[idx - 1].word in PUNCTUATION_SEPERATORS):
      lastPuncGroupIdx = idx
    if (token_list[idx].pos == 'CC'):
      lastCCGroupIdx = idx
    if idx == len(token_list) - 1:
      annotate_CC_right(idx, lastCCGroupIdx, token_list)

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

def annotate_conjunction(currentIdx, lastGroupIdx, tokens):
  if (tokens[currentIdx].pos == 'CC'):
    for idx in range(lastGroupIdx, currentIdx):
      tokens[idx].attrs[CCLEFT_F] += 1
      if (lastGroupIdx > 0):
        tokens[idx].attrs[CCRIGHT_F] += 1

def annotate_CC_right(currentIdx, lastGroupIdx, tokens):
  ''' if the lastGroupIdx is greater than 0, annotate the CCRIGHT_F for the words that need it.'''
  if (lastGroupIdx > 0):
    for idx in range(lastGroupIdx, currentIdx + 1):
      tokens[idx].attrs[CCRIGHT_F] += 1
    

#weight hashtags
def annotateHashtag(token):
  '''Take a token, and check to see if a token is a hashtag'''
  token.attrs[HASHTAG_F] = int(token.word[0] == "#" and len(token.word) > 1)

#correct or incorrect spelling as a feature
def annotateSpelling(token):
  '''Take a token, and check if is spelled correctly'''
  if ("@" not in token.word):
    spelledCorrectly = spell_dict.check(token.word)
    token.attrs[SPELL_F] = int(spelledCorrectly)

  else:
    token.attrs[SPELL_F] = 1

def annotate_and_correct_spelling(token):
  '''Takes a token, performs spell check then naively corrects spelling if mispelled'''
  if (((token.word[0] != "#" and token.word[0] != "@") and len(token.word) > 1) and token.word != 'httpstco'):
    # ignore #hashtags @mentions, 1 letter words, and our httpstco string.
    spelledCorrectly = spell_dict.check(token.word)
    token.attrs[SPELL_F] = int(spelledCorrectly)
    if (not spelledCorrectly):
      # pyenchant offers a naive spellsuggestion where index zero is "most likely".
      # print('token.word:', token.word, ' suggested:', spell_dict.suggest(token.word))
      suggestions = spell_dict.suggest(token.word)
      if (len(suggestions) > 0):
        token.word = suggestions[0]

  else:
    token.attrs[SPELL_F] = 1
  
def annotate_sentiment(token):
  '''Take a token and assign it a naive sentiment value'''
  # print('token.word: ', token.word, ' token.pos: ', token.pos)
  if token.pos in BROWN_TO_SYNSET.keys():
    breakdown = list(swn.senti_synsets(token.word, BROWN_TO_SYNSET[token.pos]))
    if (len(breakdown) > 0):
      token.attrs[POSITIVE_F] = breakdown[0].pos_score()
      token.attrs[NEGATIVE_F] = breakdown[0].neg_score()
      token.attrs[OBJECTIVE_F] = breakdown[0].obj_score()
    else:
      token.attrs[POSITIVE_F] = 0
      token.attrs[NEGATIVE_F] = 0
      token.attrs[OBJECTIVE_F] = 0
  else:
      token.attrs[POSITIVE_F] = 0
      token.attrs[NEGATIVE_F] = 0
      token.attrs[OBJECTIVE_F] = 0

