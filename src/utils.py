import os
import time
import math
import torch
import datetime
import torch.utils.data

from sklearn import metrics
from visdom import Visdom
from torch.autograd import Variable


import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data.dataset as torch_dataset

# create a variable => borrowed from jgc123
def variable(obj, volatile=False):
    if isinstance(obj, (list, tuple)):
        return [variable(o, volatile=volatile) for o in obj]

    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)

    obj = cuda(obj)
    obj = Variable(obj, volatile=volatile)
    return obj


# make cuda object
def cuda(obj):                  #  noqa
    if torch.cuda.is_available():
        obj = obj.cuda()
    return obj


def get_seq_lengths(t, pad_id):
    return torch.sort(torch.sum(torch.ne(t, pad_id), dim=1), descending=True)


def split_dataset(dataset, iter):
    val_size = 0.1 * len(dataset)
    start_idx = math.ceil(len(dataset) * (iter * .1))

    val_indices = [i for i in range(start_idx, int(start_idx+val_size))]
    train_indices = [i for i in range(start_idx)] + \
                    [i for i in range(int(start_idx + val_size), len(dataset))]

    assert len(val_indices)+len(train_indices) == len(dataset)

    val_set = torch_dataset.Subset(dataset, val_indices)
    train_set = torch_dataset.Subset(dataset, train_indices)

    return train_set, val_set


def normalize_out(output):
    y_pred = [0 if x > y else 1 for x, y in output]
    return variable(torch.Tensor(y_pred))


def get_accuracy(labels, preds):
    acc = []
    for x, y in zip(labels, preds):
        acc.append(metrics.accuracy_score(x.cpu().data.numpy(), y.cpu().data.numpy()))
        #acc.append(accuracy_score(x, y))

    return np.mean(acc)


def plot_loss(l1, l2, ac1, ac2, name, fold):
    """
    Plots train and validation losses and accuracies
    :param l1: list of train losses
    :param l2: list of validation losses
    :param ac1: list of train accuracies
    :param ac2: list of validation accuracies
    :param name: train file name
    :param fold: extension to identify plot images
    """
    plt.clf()
    viz = Visdom()
    #os.system("python -m visdom.server &")
    plt.plot(l1, 'k', label='train', linewidth=.5)
    plt.plot(l2, 'r', label='val', linewidth=.5)
    plt.plot(ac1, 'y', label='Train accuracy', linewidth=.5)
    plt.plot(ac2, 'g', label='Val accuracy', linewidth=.5)
    plt.title(name)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(loc=0)
    viz.matplot(plt)
    plt.savefig('plots/loss_fold_'+str(fold)+'.png')
    #plt.show()


def plot_stats(f1, acc):
    plt.clf()
    viz = Visdom()
    plt.plot(f1, 'r', label='F1', linewidth=.5)
    plt.plot(acc, 'b', label='Accuracy', linewidth=.5)
    plt.title("Statistics")
    plt.ylabel('Value')
    plt.xlabel('Fold')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(loc=0)
    viz.matplot(plt)
    plt.savefig('plots/final.png')


def get_f1(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred)


def get_time():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
