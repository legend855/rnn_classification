
"""
Author: Patrick Kyoyetera

"""

import utils
import torch
import logging
import torch.cuda

import numpy as np
import torch.nn as nn
import torch.utils.data as torch_data
import matplotlib.pyplot as plt

from model import RNN
from dataset import ClaimsDataset
from sklearn.metrics import accuracy_score
from logging.handlers import TimedRotatingFileHandler


def main():
    # save the model
    save_path = 'history/final_'

    # create the dataset
    filename = 'data/golden_400.csv'
    # filename = 'data/train_and_test.csv'
    data = ClaimsDataset(filename)

    vocab_size = data.vocab.__len__()
    pad_id = data.vocab.token2id.get(data.PAD)
    embedding_dim = 300
    hidden_size = 24
    batch_size = 200
    nb_epochs = 150
    lr = 1e-4
    max_norm = 5
    folds = 10
    dropout = 0.5
    criterion = nn.NLLLoss(weight=torch.Tensor([1.0, 2.2]).cuda())

    logger.info(utils.get_time())
    final_f1, final_acc = [], []
    train_f1, train_acc = [], []

    for f in range(folds):
        logger.info("Fold: " + str(f) + "\n")

        print("\nFold: #{}\n\nTraining...".format(f))
        train, test = utils.split_dataset(data, f)

        logger.info("\n")
        logger.info("Train size: "+str(len(train)))
        logger.info("Test size: " + str(len(test)))

        dl_train = torch_data.DataLoader(train, batch_size, shuffle=True)
        dl_test = torch_data.DataLoader(test, batch_size, shuffle=True)

        model = RNN(vocab_size, embedding_dim, hidden_size, pad_id, dropout, data)
        model = utils.cuda(model)
        model.zero_grad()

        parameters = list(model.parameters())
        # parameters = list(p for p in model.parameters() if p.requires_grad)

        optim = torch.optim.Adam(parameters, lr, weight_decay=35e-3, amsgrad=True)
        accuracy, f1_ep, losses = [], [], []

        for epoch in range(nb_epochs):
            model.train()
            ep_loss, output, label_list, f1 = [], [], [], []

            for _, inputs in enumerate(dl_train):
                optim.zero_grad()

                claims, labels = inputs
                labels = utils.variable(labels)

                out = model(claims)
                output.append(utils.normalize_out(out))
                label_list.append(labels)

                out = torch.log(out)
                loss = criterion(out, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, max_norm)
                optim.step()
                ep_loss.append(loss.item())

            # gather losses from single epoch
            losses.append((np.mean(ep_loss)))

            for a, b in zip(label_list, output):
                f1.append(utils.get_f1(a.data.cpu().numpy(), b.data.cpu().numpy()))
            f1 = np.mean(f1)
            acc = utils.get_accuracy(label_list, output.data.cpu().numpy())
            accuracy.append(acc)
            f1_ep.append(f1)

            print("Epoch: {} \t Accuracy: {:.3f} \t F1: {:.3f}".format(epoch, acc, f1))
        train_acc.append(np.mean(f1_ep))
        train_f1.append(np.mean(accuracy))

        # test
        print("\nTesting...")
        test_f1, test_acc = [], []
        model.eval()
        for _, test_in in enumerate(dl_test):
            cl, lab = test_in
            lab = utils.variable(lab)

            out = model(cl)
            y_pred = utils.normalize_out(out)

            test_f1.append(utils.get_f1(lab, y_pred))
            test_acc.append(accuracy_score(lab, y_pred))
        t_f1, t_acc = np.mean(test_f1), np.mean(test_acc)
        final_acc.append(t_acc)
        final_f1.append(t_f1)
        print("\nTest F1: {:.3f} \t Test Accuracy: {:.3f}".format(t_f1, t_acc))
        logger.info("F1: "+str(t_f1)+"\tAccuracy: "+str(t_acc))
    print("\nAverages...\n F1: {:.3f} \t Accuracy: {:.3f}".format(np.mean(final_f1), np.mean(final_acc)))
    logger.info("Averages: \nF1: "+str(np.mean(final_f1)) + "\t Accuracy: "+str(np.mean(final_acc)))

    utils.plot_stats(train_f1, train_acc)


if __name__ == '__main__':
    logfile = 'logs/test.log'
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    logger = logging.getLogger('Claims')
    logger.setLevel(logging.DEBUG)

    fh = TimedRotatingFileHandler(logfile, when='H', interval=1)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.info("\n")

    main()