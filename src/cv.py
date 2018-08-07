
import math
import torch
import torch.cuda
import skorch
import logging

import torch.nn as nn

from utils import cuda, variable
from model import RNN
from dataset import ClaimsDataset
from skorch import NeuralNetClassifier
from sklearn.metrics import accuracy_score


class Train(skorch.NeuralNet):
    def __init__(self, module, lr, norm *args, **kwargs):
        self.module = module
        self.lr = lr
        self.norm = norm

        super(Train,  self).__init__(self.module, *args, **kwargs)

    def initialize_optimizer(self):
        self.params = [p for p in self.module.parameters(self) if p.requires_grad]
        self.optimizer = torch.optim.Adam(params=self.params, lr=self.lr, weight_decay=35e-3, amsgrad=True)

    def train_step(self, Xi, yi, **fit_params):
        self.module.train()

        self.optimizer.zero_grad()
        yi = variable(yi)

        output = self.module(Xi)

        loss = self.criterion(output, yi)
        loss.backward()

        nn.utils.clip_grad_norm_(self.params, max_norm=self.norm)
        self.optimizer.step()

    def initialize_criterion(self):
        self.criterion = torch.nn.NLLLoss(weight=torch.Tensor([1.0, 2.0]).cuda())

    def score(self, y_t, y_p):
        return accuracy_score(y_t, y_p)

    #def infer(self, x, **fit_params):


def main():

    filename = 'data/golden_test_and_val.csv'
    embedding_dim = 300
    hidden_size = 24
    batch_size = 64
    nb_epochs = 200
    lr = 3e-4
    max_norm = 3

    #train = Train(module=RNN, criterion=nn.CrossEntropyLoss, lr=lr, norm=max_norm)


    ds = ClaimsDataset(filename)
    vocab_size = ds.vocab.__len__()
    pad_id = ds.vocab.token2id.get('<pad>')

    val_len = math.ceil(ds.__len__() * .10)
    tr_len = ds.__len__() - val_len

    # Split data
    d_tr, d_val = torch.utils.data.dataset.random_split(ds, [tr_len, val_len])
    dl_tr = torch.utils.data.DataLoader(d_tr, batch_size)
    dl_val = torch.utils.data.DataLoader(d_val, batch_size)

    model = RNN(vocab_size, hidden_size, embedding_dim, pad_id, ds)
    model = cuda(model)

    params = model.parameters()

    '''
    net = NeuralNetClassifier(module=RNN, module__vocab_size=vocab_size, module__hidden_size=hidden_size,
                              module__embedding_dim=embedding_dim, module__pad_id=pad_id,
                              module__dataset=ClaimsDataset, lr=lr, criterion=nn.CrossEntropyLoss,
                              optimizer=torch.optim.Adam, optimizer__weight_decay=35e-3, device='cuda',
                              max_epochs=nb_epochs, warm_start=True)

    '''

    trainer = Train(module=RNN, module__vocab_size=vocab_size, module__hidden_size=hidden_size,
                    module__embedding_dim=embedding_dim, module__pad_id=pad_id, lr=lr, norm=max_norm,)

    trainer.initialize(vocab_size, hidden_size, embedding_dim, pad_id)

    '''
    net.initialize()
    #net.load_params()

    # how to handle the data skorch needs to predict
    for i, inp in enumerate(dl_tr):
        claims, labels = inp
        print(len(claims), len(labels))

        net.fit(claims, labels)



    y_pred, y_true = [], []
    for i, inp in enumerate(dl_val):
        claim, label = inp
        y_true.append(variable(label))
        y_pred.append(net.predict(claim))

    acc = accuracy_score(y_true, y_pred)
    print(acc)
    '''


if __name__ == '__main__':
    main()