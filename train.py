import math
import torch
import torch.cuda
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime

from dataset import ClaimsDataset
from model import RNN
from utils import cuda, variable
from collections import defaultdict
from sklearn import metrics


def main():
    # load data
    #filename = 'data/train_and_test.csv'
    filename = 'data/data_sample.csv'

    embedding_size = 128
    hidden_size = 128
    batch_size = 64
    nb_epochs = 25
    lr = 1e-3
    max_norm = 3

    # Dataset
    ds = ClaimsDataset(filename)
    vocab_size = ds.vocab.__len__()
    pad_id = ds.vocab.token2id.get('<pad>')
    print("\nDataset size: {}".format(ds.__len__()))

    val_len = math.ceil(ds.__len__() * .15)
    test_len = batch_size
    train_len = ds.__len__() - (test_len + val_len)
    print("\nTrain size: {}\tValidate size: {}\tTest size: {}".format(train_len, val_len, test_len))

    # randomly split dataset into tr, te, & val sizes
    d_tr, d_val, d_test = torch.utils.data.dataset.random_split(ds, [train_len, val_len, test_len])

    # data loaders
    dl_tr = torch.utils.data.DataLoader(d_tr, batch_size=batch_size)#, shuffle=True)
    dl_val = torch.utils.data.DataLoader(d_val, batch_size=batch_size)#, shuffle=True)
    dl_test = torch.utils.data.DataLoader(d_test, batch_size=batch_size)

    model = RNN(vocab_size, hidden_size, embedding_size, pad_id)
    model = cuda(model)

    model.zero_grad()
    parameters = list(model.parameters())
    optim = torch.optim.Adam(parameters, lr=lr, weight_decay=0.2, amsgrad=True) # optimizer
    criterion = nn.MultiLabelSoftMarginLoss() # = nn.BCEWithLogits() # loss function

    losses = defaultdict(list)

    phases, loaders = ['train', 'val'], [dl_tr, dl_val]

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

                out = model(claim)

                labels = variable(labels.float())
                loss = criterion(out, labels)

                # back propagate, for training only
                if phase == 'train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)  # exploding gradients? say no more!
                    optim.step()

                ep_loss.append(loss.item())

            losses[phase].append(np.mean(ep_loss))  # record only average losses from single epoch

            print("Epoch: {} \t Phase: {} \t Loss: {:.4f}".format(epoch, phase, loss))

    print("\nTime finished: {}\n".format( datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')))

    plot_loss(losses['train'], losses['val'])

    # predict
    for i, inputs in enumerate(dl_test):
        claim, label = inputs
        out = model(claim)
        y_pred = normalize_out(out)

        f1 = get_f1(label, y_pred) # f1 score
        print("\n\t\tF1 score: {}\n\n".format(f1))


# my very own binary sigmoid lol
def normalize_out(output):
    y_pred = [0 if val < 0.5 else 1 for val in output]
    return y_pred


def plot_loss(l1, l2):
    plt.plot(l1, 'r', label='train', linewidth=.5)
    plt.plot(l2, 'g', label='val', linewidth=.5)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc=0)
    plt.savefig('losses.png')
    plt.show()


def get_f1(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred)


if __name__ == '__main__':
    main()
