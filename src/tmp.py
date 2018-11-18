import skorch
import torch
import utils

import torch.nn as nn

from skorch.dataset import CVSplit
from model import RNN
from skorch import NeuralNetClassifier
from dataset import ClaimsDataset
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


class Trainer(NeuralNetClassifier):
    def __init__(self, module, *args, **kwargs):
        self.module = module

        super(Trainer, self).__init__(module, *args, **kwargs)



if __name__ == '__main__':

    filename = 'data/golden_400.csv'
    testfile = 'data/test.csv'

    params = {
        'lr': [1e-4, 3e-4],
        'max_epochs': [150],
        'module__hidden_size': [20, 24, 32],
        'module__dropout': [0.3, 0.4, 0.5, 0.6],
    }

    '''
    net = Trainer(RNN, dataset=ClaimsDataset(filename),
                      train_split=torch.utils.data.dataset.random_split)
    net.initialize()
    net.fit(ClaimsDataset(filename), y=None)
    '''
    split = CVSplit(cv=0.2)
    net = NeuralNetClassifier(RNN, device='cuda', max_epochs=200, lr=1e-2,
                              module__hidden_size=24, module__vocab_size=6302,
                              module__embedding_dim=300, module__pad_id=0,
                              module__dropout=0.5, module__dataset=ClaimsDataset(filename),
                              optimizer=torch.optim.Adam, train_split=None)

    gs = GridSearchCV(net, params, cv=2, scoring='accuracy')
    net.fit(ClaimsDataset(filename), y=None)
    test = ClaimsDataset(testfile)
    y_ = test.sc['is complaint valid']
    y_true = [test.label_bin(a) for a in y_]
    y_pred = net.predict(test)

    print(accuracy_score(y_true, y_pred))

    #gs.fit(ClaimsDataset(filename), y=None)