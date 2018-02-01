import string, re
import pandas as pd
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer


stemmer = SnowballStemmer('english')

df = pd.read_json('BD_first_thousand_lines.txt', lines=True)

emoticons_str = r"""
          (?:
              [:=;] # Eyes
              [oO\-]? # Nose (optional)
              [D\)\]\(\]/\\OpP] # Mouth
              )"""


emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE)

def is_emoji(s, emoticon_re=emoticon_re):
    if re.findall(emoticon_re, s) == []:
        return False
    else:
        return True


def textPreprocessing(string, stemmer, charToKeep = string.ascii_lowercase, sWords = set(stopwords.words('english'))):
    '''Cleaning the text
    string: sentence to clean
    stemmer: stemmer to use
    charToKeep: string containing all the chars to char to keep
    sWords: stop words
    '''
    string = string.lower()
    new_string = []
    for word in string.split(' '):
        if word not in sWords:
            w = ''
            for char in word:
                if char in charToKeep:
                    w += char
            if w != '':
                new_string += [stemmer.stem(w)]
            elif word.isdigit():
                new_string += ['$DIGIT$']
            elif is_emoji(word):
                new_string += [word]

    return ' '.join(new_string)

df['reviewTextStemmed'] = df['reviewText'].apply(lambda x: textPreprocessing(x, stemmer))

cv = CountVectorizer(analyzer='word')
cv.fit(list(df['reviewTextStemmed']))
bow = cv.transform(list(df['reviewTextStemmed'])).todense()
