from gensim.models import Word2Vec
from utils import AmazonTextRecurrent, split_ids
from text2vec import textPreprocessing
from torch.utils.data import DataLoader
from torch.nn import BCELoss, MSELoss
from torch.autograd import Variable
from word2vecRNN import RNNet, evaluate
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import SGD, RMSprop, Adam
import matplotlib.pyplot as plt
plt.ion()

w2v_path = 'models/w2v_256_300_no_stem_05mars_14:38:00.model'
w2v = Word2Vec.load(w2v_path)

rnn = torch.load('models/rnn_256_300_no_stem_05mars_14:38:00.model').cuda().eval()
param = []
for n, p in rnn.named_parameters():
    if 'lstm' in n:
        p.requires_grad = False
    else:
        param += [p]

at = AmazonTextRecurrent(w2v=w2v, path='data/aclImdb/raw.csv', max_len=500, preprocessor=textPreprocessing, stemmer=None)
train_ids, train_dev_ids = split_ids(at, 10000)
train_sampler = SubsetRandomSampler(train_ids)
train_dev_sampler = SubsetRandomSampler(train_dev_ids)
dl_imdb = DataLoader(at, num_workers=4, batch_size=64, sampler=train_sampler)
dl_imdb_test = DataLoader(at, num_workers=4, batch_size=256, sampler=train_dev_sampler)
adam = Adam(params=iter(param), lr=1e-3)
loss = MSELoss()

loss_track = []
for epoch in range(5):
    for i, data in enumerate(dl_imdb):
        adam.zero_grad()
        x = Variable(data['w2v']).permute(1, 0).cuda()
        y = Variable(data['label']).permute(1, 0).cuda()
        y_hat = rnn(x).squeeze(2)
        los = loss(y_hat, y)
        los.backward()
        adam.step()
        loss_track += [los.data[0]]
    print('Epoch {} -'.format(epoch), end=' ')
    rnn.eval()
    evaluate(rnn, dl_imdb_test, loss, verbose=0)

plt.plot(loss_track)
# wrong = []
# for data in at:
#     x = Variable(data['w2v']).cuda()
#     y = Variable(data['label']).cuda()
#     y_hat = rnn(x.unsqueeze(1))
#     y_hat = (torch.mean(y_hat[-20:, 0], dim=0)
#              > .5).cpu().data.numpy().astype('float32')
#     if y[-1].data.cpu().numpy() != y_hat[0]:
#         wrong += [data['text']]
