
# Patrick Kyoyetera
# copyright 2018

from nltk import RegexpTokenizer
from nltk.corpus import stopwords 
from bs4 import BeautifulSoup
from collections import Counter 

import pandas as pd
import torch.utils.data


class Vocabulary(object):
    def __init__(self, special_tokens=None):
        """
        Make a vocabulary
        :param special_tokens:
        """
        super(Vocabulary, self).__init__()

        self.num_tokens = 0
        self.token2id, self.id2token = {}, {}
        self.token_counts = Counter()

        self.special_tokens = []
        if special_tokens is not None:
            self.special_tokens = special_tokens
            self.add_document(self.special_tokens)

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
    headers = ['title', 'complain']

    def __init__(self, csvfile):
        """
        Clean data and create a vocabulary from given csvfile
        :param csvfile: csv file with claims and labels
        """
        print("\nPreparing vocab...\n")
        self.filename = csvfile

        df = self.load_data()
        self.sc = self.tokenize_data(df)

        self.max_len_t = self.sc['title'].apply(lambda row: len(row)).max()
        self.max_len_c = self.sc['complain'].apply(lambda row: len(row)).max()

        self.make_vocab()
        self.nb_sentences = len(self.sc)
        print("Vocab ready; Size: {} tokens\n".format(len(self.vocab.token2id)))

        super(ClaimsDataset, self).__init__()

    def load_data(self):
        """
        Clan out dataset and return only the relevant columns
        :return: dataframe of columns: title, claim and label
        """
        df = pd.read_csv(self.filename)
        df = df.loc[:, ['title', 'complain', 'is complaint valid']]
        df = df.dropna()  # subset=['is complaint valid'])
        df = df.reset_index(drop=True)

        df['complain'] = df['complain'].apply(lambda item: BeautifulSoup(item, "lxml").text)

        return df

    @staticmethod
    def tokenize_data(df_):
        """
        Lower case and tokenize sentences
        :param df_: dataframe
        :return: dataframe with lower cased and tokenized sentences
        """
        tkzr = RegexpTokenizer('\w+')
        stop_words = set(stopwords.words('english'))

        # lower case, tokenize & stop words
        for header in ClaimsDataset.headers:
            df_[header] = df_[header].apply(lambda row: tkzr.tokenize(row.lower()))
            df_[header] = df_[header].apply(lambda line: [w for w in line if w not in stop_words])
        return df_

    def make_vocab(self, vocab=None):
        """
        Create the vocabulary of words found in the dataset
        :param vocab: Predefined vocabulary
        """
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = Vocabulary([ClaimsDataset.PAD, ClaimsDataset.INIT,
                                     ClaimsDataset.UNK,  ClaimsDataset.EOS])

        for header in ClaimsDataset.headers:
            self.vocab.add_documents(self.sc[header])
            self.sc[header] = self.sc[header].apply(lambda line: line[:self.max_len_t - 1]
                                                                if header == 'title'
                                                                else line[:self.max_len_c - 1])
            self.sc[header] = self.sc[header].apply(lambda line: line + [ClaimsDataset.EOS, ])

    def __getitem__(self, idx):
        """
        given an index of an item, return it's label and word ids
        :param idx: index of line
        :return: tuple of tensor of word ids in sentence and its label
        """
        if isinstance(idx, int):
            line = self.sc.iloc[idx]
        else:
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
        
        # convert to indices
        _title = [self.vocab.token2id[t] if t in self.vocab.token2id
                                         else self.vocab.token2id[ClaimsDataset.UNK] for t in _title]
        _comp = [self.vocab.token2id[t] if t in self.vocab.token2id 
                                         else self.vocab.token2id[ClaimsDataset.UNK] for t in _comp]

        # turn to torch tensors for data loader
        _title, _comp = torch.Tensor(_title), torch.Tensor(_comp)
        return _comp, _lab

    def __len__(self):
        """
        Get size of dataset
        :return: number of sentences
        """
        return self.nb_sentences

    @staticmethod
    def label_bin(s):
        if s == 'Y':
            return 1
        else:
            return 0


if __name__ == '__main__':
    d = ClaimsDataset('data/golden_400.csv')
    print(d.__getitem__(5)[0])
