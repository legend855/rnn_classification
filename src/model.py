
import torch
import torch.nn as nn

from utils import get_seq_lengths, variable
from torch.autograd import Variable
from vectors import get_embs, elmoEmbeddings


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, pad_id, dropout, dataset=None):
        super(RNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim    # 300 for pretrained, 128 for pytorch embeddings
        self.hidden_size = hidden_size
        self.pad_id = pad_id
        self.drop = dropout

        # layers
        # self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.emb = nn.Embedding.from_pretrained(get_embs(dataset), freeze=True)
        #self.emb = elmoEmbeddings(dataset)
        # self.rnn = nn.LSTM(self.embedding_dim, self.hidden_size, batch_first=True)
        self.rnn = nn.GRU(self.embedding_dim, self.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=self.drop)
        # self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, 2)
        self.sfx = nn.Softmax(dim=1)

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

        # GRU
        batch_size = inputs.shape[0]
        hidden = self.init_hidden(batch_size)
        inputs = variable(inputs)  # wrap in tensors => cuda

        lengths, indices = get_seq_lengths(inputs, self.pad_id)
        inputs = inputs[indices]

        embeds = self.emb(inputs.long())
        drop_em = self.dropout(embeds)

        # pack inputs
        packed_in = nn.utils.rnn.pack_padded_sequence(drop_em, lengths.tolist(), batch_first=True)

        # run through rnn
        out, hn = self.rnn(packed_in, hidden.unsqueeze(dim=0))

        # unpack
        unpacked, un_lengths = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # Extract the outputs at last timestep.
        ### NOTE: This is output at each sequence's original lengths
        idx = (torch.cuda.LongTensor(lengths) - 1).view(-1, 1).expand(len(lengths), unpacked.size(2))
        idx = idx.unsqueeze(1)
        mid_out = unpacked.gather(1, idx).squeeze(1)

        # output_1 = self.fc2(hn.float())
        output_1 = self.fc2(mid_out)
        #output_2 = self.fc2(self.dropout(output_1))
        output = self.sfx(output_1)

        _, unsort_ind = torch.sort(indices)  # unsort
        output = output[unsort_ind]  # dimensions

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