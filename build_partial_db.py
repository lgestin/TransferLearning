import gzip
import pandas as pd
import numpy as np

from nltk.stem import SnowballStemmer
from text2vec import textPreprocessing

np.random.seed(0)
stemmer = SnowballStemmer('english')

n_lines = 500000
n_valid = 100000
data = np.empty((n_lines, 2), dtype='object')
data_valid = np.empty((n_valid, 2), dtype='object')

n_neg = int(.4 * n_lines)
n_pos = int(.4 * n_lines)
n_2 = int(.1*n_lines)
n_4 = int(.1*n_lines)

pos = 0
neg = 0
n2 = 0
n4 = 0
nl = 0
nv = 0

with gzip.GzipFile('/media/lucas/Seagate Expansion Drive1/BD_Amazon/BD.gz') as fin:
    for i, line in enumerate(fin):
        r = np.random.random()
        s = pd.read_json(line.decode('ascii'))[['overall', 'reviewText']]
        overall = float(s['overall'][0])
        if nl < n_lines and r > n_valid / (n_lines + n_valid):
            if overall == 1 and neg < n_neg:
                data[nl, 0] = overall
                data[nl, 1] = textPreprocessing(s['reviewText'][0], stemmer)
                nl += 1
                neg += 1
            elif overall == 2 and n2 < n_2:
                data[nl, 0] = overall
                data[nl, 1] = textPreprocessing(s['reviewText'][0], stemmer)
                nl += 1
                n2 += 1
            elif overall == 5 and pos < n_pos:
                data[nl, 0] = overall
                data[nl, 1] = textPreprocessing(s['reviewText'][0], stemmer)
                nl += 1
                pos += 1
            elif overall == 4 and n4 < n_4:
                data[nl, 0] = overall
                data[nl, 1] = textPreprocessing(s['reviewText'][0], stemmer)
                nl += 1
                n4 += 1
        elif nv < n_valid and r <= n_valid / (n_lines + n_valid):
            data_valid[nv, 0] = overall
            data_valid[nv, 1] = textPreprocessing(s['reviewText'][0], stemmer)
            nv += 1
        elif nl >= n_lines and nv >= n_valid:
            break
        if (nl+nv) % 500 == 499:
            print('Train: {0:.6f}%  |  Test: {0:.6f}%'.format(str(nl/n_lines), str(nv/n_valid)), end='\r')


df = pd.DataFrame(data, columns=['overall', 'reviewTextStemmed'])
df_valid = pd.DataFrame(data_valid, columns=['overall', 'reviewTextStemmed'])
df[~df['reviewTextStemmed'].isnull()].to_csv('stemmed_amazon_500k_train.csv', sep='\t', index=False)
df_valid[~df_valid['reviewTextStemmed'].isnull()].to_csv('stemmed_amazon_500k_test.csv', sep='\t', index=False)
