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
        self.lstm1 = torch.nn.LSTM(
            self.input_size, self.hidden_size // 2, bidirectional=True)
        self.lstm2 = torch.nn.LSTM(self.hidden_size, self.hidden_size)
        self.fc1 = torch.nn.Linear(self.hidden_size, 1)

        self.d1 = torch.nn.Dropout(p=0.6)
        self.d2 = torch.nn.Dropout(p=0.6)

        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_normal(param)

    def init_hidden(self, batch_size):
        return [(Variable(torch.zeros(2, batch_size, self.hidden_size // 2)).cuda(), Variable(torch.zeros(2, batch_size, self.hidden_size // 2)).cuda()),
                (Variable(torch.zeros(1, batch_size, self.hidden_size)).cuda(), Variable(torch.zeros(1, batch_size, self.hidden_size)).cuda())]

    def forward(self, x, hidden_states=None):
        batch_size = x.size(1)
        if hidden_states is None:
            hidden_states = self.init_hidden(batch_size)
        x, hidden_state_1 = self.lstm1(x, hidden_states[0])
        x = self.d1(x)
        x, hidden_state_2 = self.lstm2(x, hidden_states[1])
        x = self.d2(x)
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
        y_hat = (torch.mean(y_hat[-20:, :], dim=0)
                 > .5).cpu().data.numpy().astype('float32')
        acc += [1 - sum(abs(y - y_hat)) / dataloader.batch_size]
        tp += sum(y * y_hat)
        tn += sum((1 - y) * (1 - y_hat))
        fp += sum((1 - y) * y_hat)
        fn += sum(y * (1 - y_hat))
        if i % int(len(dataloader) / 10) == int(len(dataloader) / 10) - 1 and verbose == 1:
            print('loss: {0:.6f} accuracy: {1:.6f} tp/(tp+fn): {2:.6f} tn/(tn+fp): {3:.6f}'.format(
                np.mean(loss_track), np.mean(acc), tp / (tp + fn), tn / (tn + fp)), end='\r')
    if verbose != -1:
        print('loss: {0:.6f} accuracy: {1:.6f} tp/(tp+fn): {2:.6f} tn/(tn+fp): {3:.6f}'.format(
            np.mean(loss_track), np.mean(acc), tp / (tp + fn), tn / (tn + fp)))
    return np.mean(loss_track), np.mean(acc)


if __name__ == '__main__':
    from torch.optim import SGD, RMSprop, Adam
    from torch.nn import BCELoss, MSELoss
    from torch.utils.data.sampler import SubsetRandomSampler
    from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
    import matplotlib.pyplot as plt
    from utils import AmazonTextRecurrent, split_ids, build_name
    import gensim
    from nltk.stem import SnowballStemmer
    from text2vec import textPreprocessing

    plt.ion()

    # Parameters
    w2v_size = 300
    max_len = 300
    n_epoch = 2
    n_hidden = 256
    stemmer = None  # SnowballStemmer('english')

    # Word2Vec
    w2v_path = None  # 'models/w2v_1M_300_no_stem.model'
    if w2v_path is None:
        df = pd.read_csv('data/raw_amazon_5M.csv', sep='\t')
        df = df[~df['reviewText'].isnull()].reset_index()
        df['reviewText'] = df['reviewText'].apply(
            lambda x: textPreprocessing(x))
        print('Fitting Word2Vec...', end=' ')
        w2v = Word2Vec([x.split(' ') for x in df['reviewText'].values],
                       size=w2v_size, window=5, min_count=100, workers=8, iter=10)
        w2v_name = build_name('w2v', w2v_size, n_hidden, stemmer)
        w2v.save(w2v_name)
        print('Done')
    else:
        w2v = Word2Vec.load(w2v_path)
        print('Word2Vec loaded.')

    # w2v = gensim.models.KeyedVectors.load_word2vec_format(
    #     'models/GoogleNews-vectors-negative300.bin', binary=True)
    w2v_size = w2v.vector_size
    # print('Word2Vec loaded.')

    model_name = build_name('rnn', w2v_size, n_hidden, stemmer)

    at = AmazonTextRecurrent(w2v=w2v, path='data/raw_amazon_5M.csv',
                             max_len=max_len, preprocessor=textPreprocessing, stemmer=stemmer)
    at_test = AmazonTextRecurrent(
        w2v=w2v, path='data/raw_amazon_5M_test.csv', max_len=max_len, preprocessor=textPreprocessing, stemmer=stemmer)
    train_ids, train_dev_ids = split_ids(at, 10000)
    test_ids, test_dev_ids = split_ids(at_test, 10000)

    train_sampler = SubsetRandomSampler(train_ids)
    train_dev_sampler = SubsetRandomSampler(train_dev_ids)
    test_sampler = SubsetRandomSampler(test_ids)
    test_dev_sampler = SubsetRandomSampler(test_dev_ids)

    dl = DataLoader(at, batch_size=64, num_workers=4,
                    shuffle=False, sampler=train_sampler)
    dl_train_dev = DataLoader(
        at, batch_size=256, shuffle=False, sampler=train_dev_sampler, num_workers=2)
    dl_test = DataLoader(at_test, batch_size=256, num_workers=2, sampler=test_sampler)
    dl_test_dev = DataLoader(at_test, batch_size=256, num_workers=2, sampler=test_dev_sampler)

    rnn = RNNet(w2v_size, n_hidden).cuda()

    # sgd = SGD(params=rnn.parameters(), lr=.9, momentum=.8)
    # rms = RMSprop(params=rnn.parameters(), lr=.0003, momentum=.5)
    adam = Adam(params=rnn.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(adam, patience=3, factor=0.5)
    loss = MSELoss()
    loss_track = []
    eval_track = []
    train_dev_track = []

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
            if i % int(len(dl) / 20) == 0:
                rnn.eval()
                eval_err, _ = evaluate(rnn, dl_test_dev, loss, verbose=-1)
                train_dev_err, _ = evaluate(rnn, dl_train_dev, loss, verbose=-1)
                if eval_track != [] and eval_err < min(eval_track):
                    torch.save(rnn, model_name)
                rnn.train()
                eval_track += [eval_err]
                train_dev_track += [train_dev_err]
                print('training error: {0:.6f}  -  eval error: {1:.6f}  -  train_dev error: {2:.6f}'.format(
                    np.mean(loss_track[-1000:]), eval_err, train_dev_err), end='\r')
                scheduler.step(eval_err)

        print('Epoch {} -'.format(epoch), end=' ')
        rnn.eval()
        evaluate(rnn, dl_test, loss, verbose=0)

    dl_imdb = DataLoader(AmazonTextRecurrent(w2v=w2v, path='data/aclImdb/raw.csv', max_len=max_len,
                                             preprocessor=textPreprocessing, stemmer=stemmer), num_workers=4, batch_size=256)
    dl_tweets = DataLoader(AmazonTextRecurrent(w2v=w2v, path='data/Tweets_test.csv',
                                               preprocessor=textPreprocessing, stemmer=stemmer), num_workers=4, batch_size=256)
    print('Tweets dataset: ', end=' ')
    evaluate(rnn, dl_tweets, loss, verbose=0)
    print('IMDB dataset  : ', end=' ')
    evaluate(rnn, dl_imdb, loss, verbose=0)
    plt.plot(np.linspace(0, n_epoch, len(loss_track)), loss_track,
             np.linspace(0, n_epoch, len(eval_track)), eval_track,
             np.linspace(0, n_epoch, len(train_dev_track)), train_dev_track)
