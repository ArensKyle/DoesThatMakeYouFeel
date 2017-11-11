import enchant, nltk, words

SPELL_F = words.new_feat()
HASHTAG_F = words.new_feat()

PERIOD_F = words.new_feat()
QUESTION_F = words.new_feat()
EXCLAMATION_F = words.new_feat()
PERIODPERIOD_F = words.new_feat()
ELLIPSIS_F = words.new_feat()

spellDict = enchant.Dict("en_US")

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
  annotate_punctuations(w_tokens)
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

def annotate_punctuations(token_list):
  ''' takes in a list of tokens and annotates the punctuation vector '''
  lastPuncGroupIdx = 0
  for idx in range(len(token_list)):
    annotatePunctuation(idx, lastPuncGroupIdx, token_list)

    #allows for the grouping of punctuation to account for repeating punctuation
    if (token_list[idx].word not in PUNCTUATION_SEPERATORS and idx > 0 and token_list[idx - 1].word in PUNCTUATION_SEPERATORS):
      # print("updating lastPuncGroup")
      lastPuncGroupIdx = idx

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
  if ("@" not in token.word):
    spelledCorrectly = spellDict.check(token.word)
    token.attrs[SPELL_F] = int(spelledCorrectly)

  else:
    token.attrs[SPELL_F] = 1

def annotate_and_correct_spelling(token):
  '''Takes a token, performs spell check then naively corrects spelling if mispelled'''
  if (((token.word[0] != "#" and token.word[0] != "@") and len(token.word) > 1) or token.word != 'httpstco'):
    # ignore #hashtags @mentions, 1 letter words, and our httpstco string.
    spelledCorrectly = spellDict.check(token.word)
    token.attrs[SPELL_F] = int(spelledCorrectly)
    if (not spelledCorrectly):
      # pyenchant offers a naive spellsuggestion where index zero is "most likely".
      # print('token.word:', token.word, ' suggested:', spellDict.suggest(token.word))
      suggestions = spellDict.suggest(token.word)
      if (len(suggestions) > 0):
        token.word = suggestions[0]

  else:
    token.attrs[SPELL_F] = 1
  
