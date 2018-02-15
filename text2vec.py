import string
import re
import pandas as pd
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

emoticons_str = r"""
          (?:
              [:=;] # Eyes
              [oO\-]? # Nose (optional)
              [D\)\]\(\]/\\OpP] # Mouth
              )"""


emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE)


def is_emoji(s, emoticon_re=emoticon_re):
    ''''check if s is an emoticon or not
    s:string
    emoticon_re:regex
    '''
    if re.findall(emoticon_re, s) == []:
        return False
    else:
        return True


def textPreprocessing(string, stemmer, charToKeep=string.ascii_lowercase, sWords=set(stopwords.words('english'))):
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
                new_string += ['DIGIT']
            elif is_emoji(word):
                new_string += [word]

    return ' '.join(new_string)


if __name__ == '__main__':

    stemmer = SnowballStemmer('english')
    print('Loading ...', end=' ')
    df = pd.read_json('BD_first_thousand_lines.txt', lines=True)[:2000]
    print('Done')

    print('Stemming...', end=' ')
    df['reviewTextStemmed'] = df['reviewText'].apply(lambda x: textPreprocessing(x, stemmer))
    print('Done')

    print('Saving new dataframe...', end=' ')
    df.reset_index().to_json('BD_first_ten_lines_stemmed.json', orient='records', lines=True)
    print('Done')

    # l = []
    # for i in range(1000, 10000, 100):
    # cv = CountVectorizer(analyzer='word', min_df = 1, max_df = 10000)
    # cv.fit(list(df['reviewTextStemmed'][:i]))
    # l += [len(cv.vocabulary_)]

    # bow = cv.transform(list(df['reviewTextStemmed'])).todense()

    # plt.plot(list(range(1000,10000,100),l)
    # plt.title('Evolution de la taille du vocabulaire\nen fonction du nombre de lignes prises')
    # plt.xlabel('Nb de ligne')
    # plt.ylabel('Taille du vocabulaire')
