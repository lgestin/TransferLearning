from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import torch


class AmazonText(Dataset):
    def __init__(self, w2v, path='balanced_stemmed_amazon_50k.csv'):
        super(AmazonText, self).__init__()
        self.w2v = w2v
        self.df = pd.read_csv(path, sep='\t')
        self.df = self.df[~self.df['reviewText'].isnull()]
        self.text = self.df['reviewText'].values
        self.text = self.text[2000:]
        self.labels = self.df['overall'].apply(
            lambda x: 0. if x <= 2.5 else 1.).values
        self.tfidf = TfidfVectorizer(
            min_df=0, lowercase=False, token_pattern=r"([^\s]+|[:=;][o0\-]?[D\)\]\(\]/\\OpP])").fit(self.text)

    def __len__(self):
        return self.text.shape[0]

    def __getitem__(self, idx):
        # TODO:Here, when unknown word is met, we replace it by zeros(which might represent a word with a meaning,
        # it is necessary to replace unknown wrds by a specific token so that representation isn't biased., to be added in preprocessing phase)
        t = self.text[idx]
        label = self.labels[idx]
        output = text2vec(t, self.w2v, tfidf=self.tfidf)
        dic = {'w2v': output, 'label': torch.Tensor([label]), 'text': t}
        return dic


class AmazonTextRecurrent(Dataset):
    def __init__(self, w2v, max_len=None, path='stemmed_amazon_500k_train.csv', preprocessor=None, stemmer=None):
        super(AmazonTextRecurrent, self).__init__()
        self.preprocessor = preprocessor
        self.stemmer = stemmer
        self.w2v = w2v
        self.df = pd.read_csv(path, sep='\t')
        self.df = self.df[~self.df['reviewText'].isnull()]
        self.df = self.df.reset_index()
        self.text = self.df['reviewText'].values
        self.labels = self.df['overall'].apply(lambda x: 0. if x <= 2.5 else 1.).values
        if max_len is None:
            self.max_len = max([len(t.split(' ')) for t in self.text])
        else:
            self.max_len = max_len

    def __len__(self):
        return self.text.shape[0]

    def __getitem__(self, idx):
        t = self.text[idx]
        label = self.labels[idx]

        if self.preprocessor is not None:
            t = self.preprocessor(t, stemmer=self.stemmer)

        output = text2matrix(t, self.w2v, max_len=self.max_len)
        dic = {'w2v': output, 'label': torch.Tensor([float(label)]*30), 'text': t}
        return dic


def text2matrix(text, w2v, max_len=None):
    text = text.split(' ')
    if max_len is None:
        max_len = len(text)
    matrix = torch.zeros((max_len, w2v.vector_size))
    i = 0
    j = 0
    while j < len(text) and i < max_len:
        try:
            matrix[i, :] = torch.Tensor(w2v[text[j]])
            i += 1
        except KeyError:
            pass
        j += 1
    return matrix


def text2vec(text, w2v, tfidf=None):
    output = torch.zeros(w2v.vector_size)
    text = text
    if tfidf is not None:
        tfidf_t = tfidf.transform([text])
        tfidfs = []
        for i, word in enumerate(text.split(' ')):
            try:
                if len(word) > 1:
                    tfidfs += [tfidf_t[0, tfidf.vocabulary_[word]]]
                else:
                    tfidfs += [0.]
            except KeyError:
                pass

    for i, word in enumerate(text.split(' ')):
        try:
            if len(word) > 1 and tfidf is not None:
                output += torch.Tensor(w2v.wv[word]) * torch.Tensor([tfidfs[i]])
            elif tfidf is None:
                output += torch.Tensor(w2v.wv[word])
        except KeyError:
            pass
    if torch.sum(output) != 0:
        output = output / torch.norm(output)
    return output


def split_ids(dataset, test_size=.2, shuffle=True, seed=0):
    length = len(dataset)
    indices = list(range(length))

    if shuffle is True:
        import random
        random.seed(seed)
        random.shuffle(indices)

    if type(test_size) is float:
        split = int(test_size * length)
    elif type(test_size) is int:
        split = test_size
    else:
        raise ValueError('%s should be an int or a float' % str)
    return indices[split:], indices[:split]


def build_name(string, w2v_size, n_hidden, stemmer):
    import time
    if stemmer is None:
        model_name = 'models/{}_{}_{}_no_stem_{}.model'.format(string, n_hidden, w2v_size, time.strftime('%d%b_%X'))
    else:
        model_name = 'models/{}_{}_{}_stem_{}.model'.format(string, n_hidden, w2v_size, time.strftime('%d%b_%X'))
    return model_name
