import torch
from gensim.models import Word2Vec
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.autograd import Variable
import pdb
from torch.nn.init import xavier_uniform

df = pd.read_csv('balanced_stemmed_amazon_350k.csv', sep='\t')
df = df[~df['reviewTextStemmed'].isnull()]

w2v_size = 2*4**4

print('Fitting Word2Vec...', end=' ')
w2v = Word2Vec([x.split(' ') for x in df['reviewTextStemmed'].values],
               size=w2v_size, window=7, min_count=0, workers=4, iter=10)
print('Done')


class FCNet(torch.nn.Module):
    def __init__(self, input_size):
        super(FCNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, int(input_size/2))
        self.fc2 = torch.nn.Linear(int(input_size/2), int(input_size/4))
        self.fc3 = torch.nn.Linear(int(input_size/4), 1)

        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_normal(param)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.sigmoid(self.fc3(x))
        return x


class AmazonText(Dataset):
    def __init__(self, w2v):
        super(AmazonText, self).__init__()
        self.w2v = w2v
        self.df = pd.read_csv('balanced_stemmed_amazon_350k.csv', sep='\t')
        self.df = self.df[~self.df['reviewTextStemmed'].isnull()]
        self.text = df['reviewTextStemmed'].values
        self.labels = df['overall'].apply(lambda x: 0. if x <= 2.5 else 1.).values
        self.tfidf = TfidfVectorizer(
            min_df=0, lowercase=False, token_pattern=r"([^\s]+|[:=;][o0\-]?[D\)\]\(\]/\\OpP])").fit(self.text)

    def __len__(self):
        return self.text.shape[0]

    def __getitem__(self, idx):
        t = self.text[idx]
        label = self.labels[idx]
        output = torch.zeros(w2v_size)
        # pdb.set_trace()
        l_t = t.split(' ')
        tfidf_t = self.tfidf.transform([t])
        # print(tfidf_t.shape)
        tfidfs = []
        # print(l_t)
        for i, word in enumerate(l_t):
            try:
                if len(word) > 1:
                    tfidfs += [tfidf_t[0, self.tfidf.vocabulary_[word]]]
                else:
                    tfidfs += [0.]
            except KeyError:
                print(word, len(word))
        for i, word in enumerate(t.split(' ')):
            if len(word) > 1:
                output += torch.Tensor(self.w2v.wv[word])*torch.Tensor([tfidfs[i]])
        # output = output.view(1, -1)
        dic = {'w2v': output/torch.norm(output), 'label': torch.Tensor([label])}
        return dic


if __name__ == '__main__':
    from torch.optim import SGD, RMSprop
    from torch.nn import BCELoss, MSELoss
    import matplotlib.pyplot as plt
    import numpy as np
    from torch.utils.data.sampler import SubsetRandomSampler
    plt.ion()

    at = AmazonText(w2v=w2v)
    train_sampler_id = []
    test_sampler_id = []
    for i in range(len(at)):
        if np.random.rand() < .2:
            test_sampler_id += [i]
        else:
            train_sampler_id += [i]
    train_sampler = SubsetRandomSampler(train_sampler_id)
    test_sampler = SubsetRandomSampler(test_sampler_id)
    dl_train = DataLoader(at, batch_size=64, shuffle=False, num_workers=4, sampler=train_sampler)
    dl_test = DataLoader(at, batch_size=len(test_sampler_id),
                         shuffle=False, num_workers=4, sampler=test_sampler)

    fcn = FCNet(w2v_size).cuda()

    sgd = SGD(params=fcn.parameters(), lr=.9, momentum=.8)
    rms = RMSprop(params=fcn.parameters(), lr=.0008, momentum=.2)
    loss = MSELoss()
    loss_track = []
    for epoch in range(20):
        for i, data in enumerate(dl_train):
            sgd.zero_grad()
            x = Variable(data['w2v']).cuda()
            y = Variable(data['label']).cuda()
            y_hat = fcn(x)
            los = loss(y_hat, y)
            los.backward()
            loss_track += [los.data[0]]
            sgd.step()
        data = next(iter(dl_test))
        x = Variable(data['w2v']).cuda()
        y = data['label'].numpy()
        y_hat = fcn(x)
        y_hat = (y_hat > .5).cpu().data.numpy().astype('float32')
        print(sum(abs(y-y_hat))/len(test_sampler_id))
