import torch
from gensim.models import Word2Vec
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.autograd import Variable
import pdb
from torch.nn.init import xavier_uniform


class FCNet(torch.nn.Module):
    def __init__(self, input_size):
        super(FCNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, int(input_size / 2))
        self.fc2 = torch.nn.Linear(int(input_size / 2), int(input_size / 4))
        self.fc3 = torch.nn.Linear(int(input_size / 4), 1)

        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_normal(param)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.sigmoid(self.fc3(x))
        return x


if __name__ == '__main__':
    from torch.optim import SGD, RMSprop
    from torch.nn import BCELoss, MSELoss
    from utils import AmazonText
    import matplotlib.pyplot as plt
    import numpy as np
    from torch.utils.data.sampler import SubsetRandomSampler
    plt.ion()

    df = pd.read_csv('balanced_stemmed_amazon_50k.csv', sep='\t')
    df = df[~df['reviewTextStemmed'].isnull()]

    w2v_size = 4**4

    # Word2Vec
    w2v_path = 'models/w2v_500k.model'
    if w2v_path is None:
        print('Fitting Word2Vec...', end=' ')
        w2v = Word2Vec([x.split(' ') for x in df['reviewTextStemmed'].values],
                       size=w2v_size, window=5, min_count=25, workers=4, iter=10)
        print('Done')
    else:
        w2v = Word2Vec.load(w2v_path)
        print('Word2Vec loaded.')

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
    dl_train = DataLoader(at, batch_size=64, shuffle=False,
                          num_workers=4, sampler=train_sampler)
    dl_test = DataLoader(at, batch_size=len(test_sampler_id),
                         shuffle=False, num_workers=4, sampler=test_sampler)
    dl_test = DataLoader(AmazonText(
        w2v=w2v, path='BD_first_thousand_lines.csv'), batch_size=2000)

    fcn = FCNet(w2v_size).cuda()

    dl = DataLoader(at, batch_size=64, num_workers=4, shuffle=True)
    sgd = SGD(params=fcn.parameters(), lr=.9, momentum=.8)
    rms = RMSprop(params=fcn.parameters(), lr=.0005, momentum=.5)
    loss = MSELoss()
    loss_track = []
    for epoch in range(20):
        for i, data in enumerate(dl):
            rms.zero_grad()
            x = Variable(data['w2v']).cuda()
            y = Variable(data['label']).cuda()
            t = data['text']
            y_hat = fcn(x)
            los = loss(y_hat, y)
            los.backward()
            loss_track += [los.data[0]]
            rms.step()
            if i % 1000 == 999:
                print(np.mean(loss_track[-100:]))
        data = next(iter(dl_test))
        x = Variable(data['w2v']).cuda()
        y = data['label'].numpy()
        y_hat = fcn(x)
        y_hat = (y_hat > .5).cpu().data.numpy().astype('float32')
        print(sum(abs(y - y_hat)) / len(y))
