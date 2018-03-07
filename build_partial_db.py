import pandas as pd
import json


def CustomParser(data, col):
    j1 = json.loads(data)
    return j1[col]


originial_path = '/media/lucas/Seagate Expansion Drive/BD_Amazon/BD'
new_path = '/media/lucas/Seagate Expansion Drive/BD_Amazon/raw_amazon_5M'

n_lines = 5000000
n_valid = 100000

df = pd.read_table(originial_path + '.gz', header=None, nrows=n_lines, compression='gzip')
for c in ['overall', 'reviewText', 'summary', 'helpful', 'reviewerID']:
    print(c)
    df[c] = df[0].apply(lambda x: CustomParser(x, c))

df = df.drop(0, axis=1)
df['binarized_overall'] = df['overall'].apply(lambda x: 0 if x < 2.5 else 1)
grouped_df = df.groupby('binarized_overall')
balanced_df = grouped_df.apply(lambda x: x.sample(grouped_df.size().min()).reset_index(drop=True))

balanced_df.drop('binarized_overall', axis=1).to_csv(new_path + '.csv', sep='\t', index=False)

df_test = pd.read_table(originial_path + '.gz', header=None, nrows=n_valid, skiprows=n_lines, compression='gzip')
for c in ['overall', 'reviewText', 'summary', 'helpful', 'reviewerID']:
    print(c)
    df_test[c] = df_test[0].apply(lambda x: CustomParser(x, c))

df_test = df_test[['overall', 'reviewText', 'summary', 'helpful', 'reviewerID']]
df_test.to_csv(new_path + '_test.csv', sep='\t', index=False)
