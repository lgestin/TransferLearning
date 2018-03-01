import torch
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pdb


class RNNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm1 = torch.nn.LSTM(self.input_size, self.hidden_size // 2, bidirectional=True)
        self.lstm2 = torch.nn.LSTM(self.hidden_size, self.hidden_size)
        self.fc1 = torch.nn.Linear(self.hidden_size, 1)

        # self.d1 = torch.nn.Dropout(p=0.5)
        # self.d2 = torch.nn.Dropout(p=0.5)

        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform(param)

    def init_hidden(self, batch_size):
        return [(Variable(torch.zeros(2, batch_size, self.hidden_size // 2)).cuda(), Variable(torch.zeros(2, batch_size, self.hidden_size // 2)).cuda()),
                (Variable(torch.zeros(1, batch_size, self.hidden_size)).cuda(), Variable(torch.zeros(1, batch_size, self.hidden_size)).cuda())]

    def forward(self, x, hidden_states=None):
        batch_size = x.size(1)
        if hidden_states is None:
            hidden_states = self.init_hidden(batch_size)
        x, hidden_state_1 = self.lstm1(x, hidden_states[0])
        # x = self.d1(x)
        x, hidden_state_2 = self.lstm2(x, hidden_states[1])
        # x = self.d2(x)
        x = torch.nn.functional.sigmoid(self.fc1(x[-30:, :, :]))
        return x


def evaluate(model, dataloader, loss, verbose=1):
    acc = []
    loss_track = []
    tp, tn, fp, fn = 0, 0, 0, 0
    for i, data in enumerate(dataloader):
        x = Variable(data['w2v']).permute(1, 0, 2).cuda()
        y = Variable(data['label']).permute(1, 0).cuda()
        y_hat = model(x).squeeze(2)
        loss_track += [loss(y_hat, y).data[0]]
        y = y[-1, :].cpu().data.numpy()
        y_hat = (torch.mean(y_hat[-20:, :], dim=0) > .5).cpu().data.numpy().astype('float32')
        acc += [1 - sum(abs(y - y_hat)) / dataloader.batch_size]
        tp += sum(y*y_hat)
        tn += sum((1-y)*(1-y_hat))
        fp += sum((1-y)*y_hat)
        fn += sum(y*(1-y_hat))
        if i % dataloader.batch_size * int(len(dataloader)/10) == dataloader.batch_size * int(len(dataloader)/10)-1 and verbose == 1:
            print('loss: {0:.6f} accuracy: {1:.6f} tp/(tp+fn): {2:.6f} tn/(tn+fp): {3:.6f}'.format(np.mean(loss_track), np.mean(acc), tp/(tp+fn), tn/(tn+fp)), end='\r')

    print('loss: {0:.6f} accuracy: {1:.6f} tp/(tp+fn): {2:.6f} tn/(tn+fp): {3:.6f}'.format(np.mean(loss_track), np.mean(acc), tp/(tp+fn), tn/(tn+fp)))
    return np.mean(loss_track), np.mean(acc)


if __name__ == '__main__':
    from torch.optim import SGD, RMSprop, Adam
    from torch.nn import BCELoss, MSELoss
    from torch.optim.lr_scheduler import StepLR
    import matplotlib.pyplot as plt
    from utils import AmazonTextRecurrent

    plt.ion()

    df = pd.read_csv('stemmed_amazon_500k_train.csv', sep='\t')
    df = df[~df['reviewTextStemmed'].isnull()]

    # Parameters
    w2v_size = 4**4
    max_len = 300
    n_epoch = 10
    n_hidden = 160

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

    at = AmazonTextRecurrent(w2v=w2v, max_len=max_len)
    dl = DataLoader(at, batch_size=64, num_workers=4, shuffle=True)
    dl_test = DataLoader(AmazonTextRecurrent(
        w2v=w2v, path='stemmed_amazon_500k_test.csv', max_len=max_len), batch_size=64)

    rnn = RNNet(w2v_size, n_hidden).cuda()

    # sgd = SGD(params=rnn.parameters(), lr=.9, momentum=.8)
    # rms = RMSprop(params=rnn.parameters(), lr=.0003, momentum=.5)
    adam = Adam(params=rnn.parameters())
    scheduler = StepLR(adam, 1, gamma=0.4)
    loss = MSELoss()
    loss_track = []
    for epoch in range(n_epoch):
        rnn.train()
        for i, data in enumerate(dl):
            adam.zero_grad()
            x = Variable(data['w2v']).permute(1, 0, 2).cuda()
            y = Variable(data['label']).permute(1, 0).cuda()
            y_hat = rnn(x).squeeze(2)
            los = loss(y_hat, y)
            los.backward()
            adam.step()
            loss_track += [los.data[0]]
            if i % 1000 == 999:
                print(np.mean(loss_track[-1000:]), end='\r')
        print('Epoch {} -'.format(epoch), end=' ')
        rnn.eval()
        evaluate(rnn, dl_test, loss, verbose=0)
        scheduler.step()
