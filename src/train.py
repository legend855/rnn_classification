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
from visdom import Visdom


def main():
    # input file
    #filename = '~/Documents/Bread/data/train_and_test.csv'
    filename = '~/Documents/Bread/data/golden_test_and_val.csv'

    embedding_size = 300
    hidden_size = 32
    batch_size = 64
    nb_epochs = 256
    lr = 1e-4
    max_norm = 3

    # Dataset
    ds = ClaimsDataset(filename)
    vocab_size = ds.vocab.__len__()
    pad_id = ds.vocab.token2id.get('<pad>')
    print("\nDataset size: {}\tVocab Size: {}".format(ds.__len__(), vocab_size))

    test_len = val_len = math.ceil(ds.__len__() * .15)
    train_len = ds.__len__() - val_len # + test_len)
    print("\nTrain size: {}\tValidate size: {}\tTest Size: {}".format(train_len, val_len, test_len))

    # randomly split dataset into tr, te, & val sizes
    d_tr, d_val = torch.utils.data.dataset.random_split(ds, [train_len, val_len])

    # data loaders
    dl_tr = torch.utils.data.DataLoader(d_tr, batch_size=batch_size)    #, shuffle=True)
    dl_val = torch.utils.data.DataLoader(d_val, batch_size=batch_size)  #, shuffle=True)
    #dl_test = torch.utils.data.DataLoader(d_te, batch_size=batch_size)

    model = RNN(vocab_size, hidden_size, embedding_size, pad_id)
    model = cuda(model)

    model.zero_grad()
    parameters = list([parameter for parameter in model.parameters() if parameter.requires_grad])
    optim = torch.optim.Adam(parameters, lr=lr, weight_decay=35e-3, amsgrad=True) # optimizer

    criterion = nn.NLLLoss() # weight=torch.Tensor([1.0, 2.0]).cuda())  # loss function

    losses = defaultdict(list)
    phases, loaders = ['train', 'val'], [dl_tr, dl_val]

    tr_acc, v_acc = [], []

    print("\nBegin training: {}\n".format( datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S') ))
    for epoch in range(nb_epochs):
        for phase, loader in zip(phases, loaders):
            if phase == 'train':
                model.train()
            else:
                model.eval()

            ep_loss, out_list, label_list = [], [], []
            for i, inputs in enumerate(loader):
                optim.zero_grad()

                claim, labels = inputs
                labels = variable(labels)

                out = model(claim)

                out_list.append(normalize_out(out))  # collect output from every epoch
                label_list.append(labels)

                out = torch.log(out)

                # criterion.weight = get_weights(labels)
                loss = criterion(out, labels)

                # back propagate, for training only
                if phase == 'train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)  # exploding gradients? say no more!
                    optim.step()

                ep_loss.append(loss.item())

            losses[phase].append(np.mean(ep_loss))  # record only average losses from every phase at each epoch

            acc = get_accuracy(label_list, out_list)
            if phase == 'train':
                tr_acc.append(acc)
            else:
                v_acc.append(acc)

            print("Epoch: {} \t Phase: {} \t Loss: {:.4f} \t Accuracy: {:.3f}".format(epoch, phase, loss, acc))

    print("\nTime finished: {}\n".format( datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')))

    plot_loss(losses['train'], losses['val'], tr_acc, v_acc, optim.param_groups[0]['weight_decay'], model.dropout.p, hidden_size)

    '''
    # predict
    f1_test = []
    for i, inputs in enumerate(dl_test):
        claim, label = inputs
        label = variable(label.float())

        out = model(claim)
        y_pred = normalize_out(out)

        #print("\n\t\tF1 score: {}\n\n".format(get_f1(label, y_pred)))   # f1 score
        f1_test.append(get_f1(label, y_pred))
    print("\t\tF1: ".format(np.mean(f1_test)))
    '''


# my very own binary sigmoid lol
def normalize_out(output):
    y_pred = [0 if x > y else 1 for x, y in output]
    return variable(torch.tensor(y_pred))


def get_accuracy(labels, preds):
    acc = []
    for x, y in zip(labels, preds):
        acc.append(metrics.accuracy_score(x, y))

    return np.mean(acc)


def plot_loss(l1, l2, ac1, ac2, r, p, h):
    viz = Visdom()

    plt.plot(l1, 'k', label='train', linewidth=.5)
    plt.plot(l2, 'r', label='val', linewidth=.5)
    plt.plot(ac1, 'y', label='Train accuracy', linewidth=.5)
    plt.plot(ac2, 'g', label='Val accuracy', linewidth=.5)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(loc=0)
    plt.savefig('losses/loss_'+'r='+str(r)+'p='+str(p)+'h='+str(h)+'.png')
    #plt.show()
    viz.matplot(plt)


def get_f1(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred)



if __name__ == '__main__':
    main()
