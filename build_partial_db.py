import gzip
import pandas as pd
import numpy as np

from nltk.stem import SnowballStemmer
from text2vec import textPreprocessing


stemmer = SnowballStemmer('english')

n_lines = 350000
data = np.empty((n_lines, 2), dtype='object')

n_neg = int(.4*n_lines)
n_pos = int(.4*n_lines)
n_2 = int(.1*n_lines)
n_4 = int(.1*n_lines)

pos = 0
neg = 0
n2 = 0
n4 = 0
nl = 0

with open('BD_partielle.csv', 'w') as fout:
    with gzip.GzipFile('/media/lucas/Seagate Expansion Drive1/BD_Amazon/BD.gz') as fin:
        for i, line in enumerate(fin):
            if nl > n_lines-1:
                break
            else:
                fout.write(line.decode('ascii'))
                s = pd.read_json(line.decode('ascii'))[['overall', 'reviewText']]
                overall = float(s['overall'][0])
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

df = pd.DataFrame(data, columns=['overall', 'reviewTextStemmed'])
df[~df['reviewTextStemmed'].isnull()].to_csv('balanced_stemmed_amazon_350k.csv', sep='\t', index=False)
