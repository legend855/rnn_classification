
import torch
import torch.nn as nn
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(self, vocab_size, hidden_size, batch_size, embedding_dim):
        super(Model, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        
        # embedding layer 
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.rnn = nn.RNN(self.embedding_dim, self.hidden_size)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, inputs):
        """
        Perform forward pass
        :param inputs
        :return: output
        """
        embs = self.emb(inputs)
        embs = self.dropout(embs)
        outputs, _ = self.rnn(embs)
        output = self.fc(outputs)
        # output = self.softmax(output)

        return output

    # def init_hidden(self):

    def cell_zero_state(self, batch_size):
        """
        Initial hidden state
        :param batch_size:
        :return: tensor of zeros of shape (batch_size x hidden_size)
        """
        weight = next(self.parameters()).data
        hidden = Variable(weight.new(batch_size, self.hidden_size).zero_())
        return hidden