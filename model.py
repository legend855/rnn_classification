
import torch
import torch.nn as nn

from utils import get_seq_lengths, variable
from torch.autograd import Variable
from vectors import get_embs


class RNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, pad_id):
        super(RNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.pad_id = pad_id
        
        # layers
        #self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.emb = nn.Embedding.from_pretrained(get_embs())
        #self.rnn = nn.LSTM(self.embedding_dim, self.hidden_size, batch_first=True)
        self.rnn = nn.GRU(self.embedding_dim, self.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.hidden_size, 2)
        self.sfx = nn.Softmax(dim=2)

    def forward(self, inputs):
        """
        Perform forward pass
        :param inputs
        :return: output
        """
        '''
        # LSTM Code
        inputs = variable(inputs) # wrap in tensors => cuda
        

        lengths, indices = get_seq_lengths(inputs, self.pad_id)
        inputs = inputs[indices]

        # embed inputs then dropout
        embeds = self.emb(inputs)
        drop_em = self.dropout(embeds)

        # pack inputs
        packed_in = nn.utils.rnn.pack_padded_sequence(drop_em, lengths.tolist(), batch_first=True)

        _, (hidden, cell) = self.rnn(packed_in)   # forward
        output = self.fc(hidden)
        output = self.sfx(output)

        _, unsort_ind = torch.sort(indices)     # unsort
        output = output.squeeze()[unsort_ind]  # dimensions
        '''
        batch_size = inputs.shape[0]
        inputs = variable(inputs)  # wrap in tensors => cuda

        lengths, indices = get_seq_lengths(inputs, self.pad_id)
        inputs = inputs[indices]

        embeds = self.emb(inputs)
        drop_em = self.dropout(embeds)

        # pack inputs
        packed_in = nn.utils.rnn.pack_padded_sequence(drop_em, lengths.tolist(), batch_first=True)

        hidden = self.init_hidden(batch_size)

        _, hn = self.rnn(packed_in, hidden.unsqueeze(dim=0))

        output = self.fc(hn.float())
        output = self.sfx(output)

        _, unsort_ind = torch.sort(indices)  # unsort
        output = output.squeeze()[unsort_ind]  # dimensions

        return output

    def init_hidden(self, batch_size):
        """
        Create initial hidden state of zeros
        :param batch_size: size of every batch
        :return: tensor of size batch_size x hidden_size of zeros
        """
        weight = next(self.parameters()).data
        hidden = Variable(weight.new(batch_size, self.hidden_size).zero_())
        return hidden
