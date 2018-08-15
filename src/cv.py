

import torch
import torch.cuda
import skorch
import logging
import torch.utils.data
import utils

import numpy as np
import torch.nn as nn
import torch.utils.data as torch_data

from model import RNN
from dataset import ClaimsDataset
from collections import defaultdict
from sklearn import metrics
from logging.handlers import TimedRotatingFileHandler


def main():

    # saved model path
    save_path = 'history/model_fold_'

    test_file = 'data/test120.csv'
    # create dataset
    #filename = 'data/golden_400.csv'
    #filename = 'data/golden_train_and_val.csv'
    filename = 'data/train_val120.csv'
    ds = ClaimsDataset(filename)
    vocab_size = ds.vocab.__len__()
    pad_id = ds.vocab.token2id.get('<pad>')

    embedding_size = 128    # 128 for torch embeddings, 300 for pre-trained
    hidden_size = 24
    batch_size = 64
    nb_epochs = 150
    lr = 1e-4
    max_norm = 5
    folds = 10
    criterion = nn.NLLLoss(weight=torch.Tensor([1.0, 2.2]).cuda())

    # For testing phase
    fold_scores = {}
    test_set = ClaimsDataset(test_file)
    dl_test = torch_data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    mean = []       # holds the mean validation accuracy of every fold
    print("\nTraining\n")
    logger.info(utils.get_time())

    for i in range(folds):
        print("\nFold: {}\n".format(i))

        losses = defaultdict(list)
        train, val = utils.split_dataset(ds, i)

        print("Train size: {} \t Validate size: {}".format(len(train), len(val)))

        dl_train = torch_data.DataLoader(train, batch_size=batch_size, shuffle=True)
        dl_val = torch_data.DataLoader(val, batch_size=batch_size, shuffle=True)

        model = RNN(vocab_size, embedding_size, hidden_size, pad_id, ds)
        model = utils.cuda(model)
        model.zero_grad()

        # When using pre-trained embeddings, uncomment below otherwise, use the second statement
        #parameters = list([parameter for parameter in model.parameters()
        #                   if parameter.requires_grad])
        parameters = list(model.parameters())

        optim = torch.optim.Adam(parameters, lr=lr, weight_decay=35e-3, amsgrad=True)

        phases, loaders = ['train', 'val'], [dl_train, dl_val]
        tr_acc, v_acc = [], []

        for epoch in range(nb_epochs):
            for p, loader in zip(phases, loaders):
                if p == 'train':
                    model.train()
                else:
                    model.eval()

                ep_loss, out_list, label_list = [], [], []
                for _, inputs in enumerate(loader):
                    optim.zero_grad()

                    claim, labels = inputs
                    labels = utils.variable(labels)

                    out = model(claim)

                    out_list.append(utils.normalize_out(out))
                    label_list.append(labels)

                    out = torch.log(out)
                    loss = criterion(out, labels)

                    if p == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
                        optim.step()

                    ep_loss.append(loss.item())
                losses[p].append(np.mean(ep_loss))

                acc = utils.get_accuracy(label_list, out_list)
                if p == 'train':
                    tr_acc.append(acc)
                else:
                    v_acc.append(acc)
                print("Epoch: {} \t Phase: {} \t Loss: {:.4f} \t Accuracy: {:.3f}"
                      .format(epoch, p, loss, acc))

        utils.plot_loss(losses['train'], losses['val'], tr_acc, v_acc, filename, i)
        mean.append(np.mean(v_acc))
        logger.info("\n Fold: "+str(i))
        logger.info("Train file=> " + filename + "\nParameters=> \nBatch size: " + str(batch_size) +
                     "\nHidden size: " + str(hidden_size) + "\nMax_norm: " + str(max_norm) +
                     "\nL2 Reg/weight decay: " + str(optim.param_groups[0]['weight_decay']) +
                     "\nLoss function: " + str(criterion))
        logger.info('Final train accuracy: ' + str(tr_acc[-1]))
        logger.info('Final validation accuracy: ' + str(v_acc[-1]))

        # Save model for current fold
        torch.save(model.state_dict(), save_path+str(i))


        test_f1, test_acc = [], []
        for _, inp in enumerate(dl_test):
            claim, label = inp
            label = utils.variable(label)

            model.eval()
            out = model(claim)
            y_pred = utils.normalize_out(out)

            test_f1.append(utils.get_f1(label, y_pred))
            test_acc.append(metrics.accuracy_score(label, y_pred))
        t_f1, t_acc = np.mean(test_f1), np.mean(test_acc)
        fold_scores[i] = dict([('F1', t_f1), ('Accuracy', t_acc)])
        print("\tf1: {:.3f} \t accuracy: {:.3f}".format(t_f1, t_acc))
        #logger.info('\nTest f1: '+str(t_f1)+'\nTest Accuracy: '+str(t_acc))

    logger.info('Mean accuracy over 10 folds: \t' + str(np.mean(mean)))
    logger.info(fold_scores)



if __name__ == '__main__':
    logfile = 'logs/cv.log'
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    logger = logging.getLogger('Claims')
    logger.setLevel(logging.DEBUG)

    fh = TimedRotatingFileHandler(logfile, when='H')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    main()



'''
class Trainer(NeuralNetClassifier):
    def __init__(self, module, *args, **kwargs):
        self.module = module

        super(Trainer, self).__init__(module, *args, **kwargs)


net = Trainer(RNN, dataset=ClaimsDataset(filename),
                  train_split=torch.utils.data.dataset.random_split)
net.initialize()
net.fit(ClaimsDataset(filename), y=None)

'''
