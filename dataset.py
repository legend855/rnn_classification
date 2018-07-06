
# Patrick Kyoyetera
# copyright 2018

from nltk import word_tokenize, sent_tokenize 
from nltk.corpus import stopwords 

from bs4 import BeautifulSoup
from collections import Counter 
from torch.autograd import Variable 
from utils import cuda, variable 

import pandas as pd 
import numpy as np 

import torch.utils.data
import contractions 


class Vocabulary(object):
    def __init__(self, special_tokens=None):
        super(Vocabulary, self).__init__()

        self.num_tokens = 0
        self.token2id = {}
        self.id2token = {}
        
        self.token_counts = Counter()

        self.special_tokens = []
        if special_tokens is not None:
            self.special_tokens = special_tokens
            self.add_document( self.special_tokens )

    # Here goes the good stuff 
    def add_document(self, doc):
        for token in doc:
            self.token_counts[token] += 1 

            if token not in self.token2id:
                self.token2id[token] = self.num_tokens
                self.id2token[self.num_tokens] = token

                self.num_tokens += 1
    
    def add_documents(self, documents):
        for line in documents:
            self.add_document(line)

    def __getitem__(self, idx):
        return self.token2id[idx]

    def __len__(self):
        return self.num_tokens


class ClaimsDataset(torch.utils.data.Dataset):
    INIT = '<sos>'
    PAD = '<pad>'
    UNK = '<unk>'
    EOS = '<eos>'

    def __init__(self, csvfile):
        self.filename = csvfile
        
        df = self.load_data()
        self.sc = self.tokenize_data(df)
        self.make_vocab()

        super().__init__()

    def load_data(self):
        df = pd.read_csv(self.filename)
        df = df.loc[:, ['title', 'complain', 'is complaint valid']]
        df = df.dropna()  # subset=['is complaint valid'])
        df = df.reset_index(drop=True)

        df['complain'] = df['complain'].apply(lambda item : BeautifulSoup(item ,"lxml").text)

        return df

    @staticmethod
    def tokenize_data(df_):
        # lower case then tokenize
        df_['title'] = df_.apply(lambda row: word_tokenize( row['title'].lower() )
                                                if type( row['title'] ) is str
                                                else word_tokenize( row['title'] ), axis=1)

        df_['complain'] = df_.apply(lambda row: word_tokenize( row['complain'].lower() )
                                                if type( row['complain'] ) is str
                                                else word_tokenize( row['complain'] ), axis=1)

        # stop words
        stop_words = set(stopwords.words('english'))
        df_['complain'] = df_['complain'].apply(lambda line : [w for w in line if w not in stop_words] )

        return df_

    def make_vocab(self, max_len_t=20, max_len_c=280, vocab=None):
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = Vocabulary( [ ClaimsDataset.PAD, ClaimsDataset.INIT, 
                                       ClaimsDataset.UNK,  ClaimsDataset.EOS ] )
        
        self.vocab.add_documents(self.sc['complain'])
        self.vocab.add_documents(self.sc['title'])
        
        self.max_len_c = max_len_c
        self.max_len_t = max_len_t 

        # Cut to max length here and append eos token
        self.sc['title'] = self.sc['title'].apply(lambda line : line[:max_len_t - 1] )
        self.sc['complain'] = self.sc['complain'].apply(lambda line : line[:max_len_c - 1] )

        self.sc['title'] = self.sc['title'].apply(lambda line : line + [ClaimsDataset.EOS, ] )
        self.sc['complain'] = self.sc['complain'].apply(lambda line : line + [ClaimsDataset.EOS, ])

        self.nb_sentences = len(self.sc)

    def __getitem__(self, idx):

        #print(type(idx))
        #print(idx.shape)
        #print(idx.item())
        line = self.sc.iloc[idx.item()]
        _title, _comp, _lab = line['title'], line['complain'], line['is complaint valid']

        _lab = self.label_bin(_lab)

        # pad title
        t_pads = self.max_len_t - len(_title)
        if t_pads > 0:
            _title = _title + [ClaimsDataset.PAD] * t_pads
        
        # pad complaint
        c_pads = self.max_len_c - len(_comp)
        if c_pads > 0:
            _comp = _comp + [ClaimsDataset.PAD] * c_pads
        
        # convert both to indices 
        _title = [self.vocab.token2id[t] if t in self.vocab.token2id
                                         else self.vocab.token2id[ClaimsDataset.UNK] for t in _title ]
        _comp = [self.vocab.token2id[t] if t in self.vocab.token2id 
                                         else self.vocab.token2id[ClaimsDataset.UNK] for t in _comp ]

        # turn to torch tensors so dataloader can handle here 

        _title, _comp = torch.tensor(_title), torch.tensor(_comp)
        
        return [_comp, _lab]

    def __len__(self):
        return self.nb_sentences

    def label_bin(self, s):
        if s == 'Y':
            return 1
        else:
            return 0



if __name__ == '__main__':

    d = ClaimsDataset('data/data_sample.csv')
    #print(type(d))
    print(d.__getitem__(2))
    #print( d.__len__() )
