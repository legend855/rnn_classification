import numpy as np
import torch
import pickle
from gensim.models import KeyedVectors
from collections import defaultdict
from dataset import ClaimsDataset


def load_embeddings(name, dd):
    """
        :param: name of embeddings .vec file, dataset object
        :type: string, ClaimsDataset object
        :return: name of pickle file where tokens and embeddings are stored
        :rtype: str
    """
    word_vec_dict = {}
    count = 0
    vocab = dd.vocab.token_counts.keys()
    outfile = 'data/pickle/vectors.pkl'

    print("Size of vocab: {}\n".format(len(vocab)))
    print("Preparing embeddings...\n")
    with open(name, 'r') as f, open(outfile, 'wb') as out:
        next(f)
        for line in f:
            line = line.split(' ')
            word = line[0]
            if word in vocab:
                word_vec_dict[word] = np.array([float(a.rstrip()) for a in line[1:]])
                count += 1
            else:
                pass

        print("Embeddings found: {}".format(count)) # number of words in Claims vocab whose embeddings now exist
        print("{} words do not have embeddings\n\n".format(len(vocab) - count))
        pickle.dump(word_vec_dict, out)
        out.close()

    return outfile


def sort_embs():
    # filename = 'data/Embeddings/wiki-news-300d-1M-subword.vec'
    filename = 'data/Embeddings/crawl-300d-2M.vec'

    dd = ClaimsDataset('data/golden_400.csv')

    infile = load_embeddings(filename, dd)

    with open(infile, 'rb') as vec_file:
        vecs = pickle.load(vec_file)    # vecs is a dictionary where key: word(str), embedding(np.array)

    non_words = [w for w in dd.vocab.token_counts.keys() if w not in vecs.keys()]
    for word in non_words:
        id = dd.vocab.token2id[word]
        dd.vocab.id2token[id] = dd.UNK
        dd.vocab.token2id.pop(word)
        dd.vocab.token2id.update()



    print("Echo")



sort_embs()


