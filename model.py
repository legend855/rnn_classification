
import torch
import torch.nn as nn
from utils import get_seq_lengths, variable


class RNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, pad_id):
        super(RNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.pad_id = pad_id
        
        # layers
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        #self.rnn = nn.RNN(self.embedding_dim, self.hidden_size)
        self.rnn = nn.LSTM(self.embedding_dim, self.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, inputs):
        """
        Perform forward pass
        :param inputs
        :return: output
        """
        inputs = variable(inputs) # wrap in tensors => cuda

        lengths, indices = get_seq_lengths(inputs, self.pad_id)
        inputs = inputs[indices]

        # pack inputs
        packed_in = nn.utils.rnn.pack_padded_sequence(self.emb(inputs), lengths.tolist(), batch_first=True)
        packed_in = self.dropout(packed_in)
        outputs, (hidden, cell) = self.rnn(packed_in)   # forward

        output = self.fc(hidden)

        _, unsort_ind = torch.sort(indices)     # unsort
        output = output.squeeze()[unsort_ind]  # dimensions

        return output
