import math

import torch
import torch.cuda
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime

from dataset import ClaimsDataset, variable
from model import RNN
from utils import cuda
from collections import defaultdict
from sklearn import metrics


def main():
    embedding_size = 128
    hidden_size = 512
    batch_size = 64
    nb_epochs = 250
    lr = 1e-3
    max_norm = 5

    # load data
    #filename = 'data/train_and_test.csv'
    filename = 'data/data_sample.csv'
    ds = ClaimsDataset(filename)

    print("\nDataset size: {}".format(ds.__len__()))

    val_len = math.ceil(ds.__len__() * .25)
    train_len = ds.__len__() - val_len
    print("\nTrain size: {}\tValidate size: {}".format(train_len, val_len))

    d_tr, d_val = torch.utils.data.dataset.random_split(ds, [train_len, val_len])

    dl_tr = torch.utils.data.DataLoader(d_tr, batch_size=batch_size)
    dl_val = torch.utils.data.DataLoader(d_val, batch_size=batch_size)

    vocab_size = ds.vocab.__len__()

    model = RNN(vocab_size, hidden_size, batch_size, embedding_size)
    model = cuda(model)
    #model.train()

    model.zero_grad()
    parameters = list(model.parameters())
    optim = torch.optim.Adam(parameters, lr=lr, weight_decay=0.6, amsgrad=True)
    criterion = nn.CrossEntropyLoss()

    losses = defaultdict(list)

    phases = ['train', 'val']
    loaders = [dl_tr, dl_val]

    print("\nBegin training: {}\n".format( datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S') ))
    for epoch in range(nb_epochs):
        for phase, loader in zip(phases, loaders):
            if phase == 'train':
                model.train()
            else:
                model.eval()

            ep_loss = []
            for i, inputs in enumerate(loader):
                optim.zero_grad()

                claim, labels = inputs
                claim, labels = variable(claim), variable(labels)

                out = model(claim)
                out = torch.squeeze(out)

                loss = criterion(out, labels)

                # back propagate, step: while training
                if phase == 'train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
                    optim.step()

                    #ep_loss.append(loss.item())
                    #print("Append train")
                ep_loss.append(loss.item())

            mean_loss = np.mean(ep_loss)
            losses[phase].append(mean_loss)

            print("Epoch: {} \t Phase: {} \t Loss: {:.4f}".format(epoch, phase, loss))# get_f1(y_t, y_p)))

    print("\nTime finished: {}".format( datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')))
    #print(losses)
    plot_loss(losses['train'], losses['val'])


def plot_loss(l1, l2):
    plt.plot(l1, 'r', label='Train', linewidth=.5)
    plt.plot(l2, 'g', label='Val', linewidth=.5)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('losses.png')
    plt.show()


def get_f1(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred)


if __name__ == '__main__':
    main()
