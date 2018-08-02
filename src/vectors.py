import numpy as np
import pickle
import os
import torch

from tqdm import tqdm


def load_embeddings(name, outfile, dd):
    """
    Open embeddings file and embeddings for words in vocab. Create a
    pickle file and store embeddings in it
        :param: name of embeddings .vec file, dataset object
        :type: string, ClaimsDataset object
    """
    word_vec_dict = {}
    count = 0
    vocab = dd.vocab.token_counts.keys()

    print("\n\nPreparing embeddings...\n")
    with open(name, 'r') as f, open(outfile, 'wb') as out:
        next(f)
        for line in tqdm(f):
            line = line.split(' ')
            word = line[0]
            if word in vocab:
                word_vec_dict[word] = np.array([float(a.rstrip()) for a in line[1:]])
                count += 1
            else:
                pass

        print("\nEmbeddings found for {} words.".format(count), end=' ') # number of words in Claims vocab whose embeddings now exist
        print("{} words do not have embeddings.\n\n".format(len(vocab) - count))
        pickle.dump(word_vec_dict, out)


def get_embs(dataset):
    """
    Add embeddings for unknown words in vocab
    :return: sorted stack of embeddings
    """
    # filename = 'data/Embeddings/wiki-news-300d-1M-subword.vec'
    raw_embedding_file = 'data/Embeddings/crawl-300d-2M.vec'
    pickled_vecs = 'data/pickle/vectors.pkl'

    if not os.path.exists(pickled_vecs):
        load_embeddings(raw_embedding_file, pickled_vecs, dataset)

    with open(pickled_vecs, 'rb') as vec_file:
        vectors = pickle.load(vec_file)  # vecs is a dictionary where key: word(str), embedding(np.array)

    # list of words that are not in current vocab
    non_words = [w for w in dataset.vocab.token_counts.keys() if w not in vectors.keys()]
    non_words = non_words[4:]   # remove special tokens
    print("\nSample words without embeddings: {}\n".format(non_words[:2]))

    # embedding for unknown tokens
    unk_emb = np.random.uniform(-1, 1, (300,))

    for word in non_words:
        vectors.update({word: unk_emb})

    with open(pickled_vecs, 'wb') as vec_file:
        pickle.dump(vectors, vec_file)

    return sort_embeddings(dataset.vocab.token2id, vectors) # START HERE TOMORROW SEND EMBEDDINGS TO ENMBEDDING LAYER


def sort_embeddings(token2id, raw_dict):
    """
    Return a stack on embeddings sorted in order as in dd.vocab
    :param token2id: dict from dd.vocab
    :param raw_dict: dictionary of embeddings pre-sorting
    :return: sequence of arrays sorted & stacked
    """
    sorted_embs = []

    for tok in token2id.keys():
        if tok in raw_dict.keys():
            sorted_embs.append(raw_dict[tok])
        else:
            # get one sort of embedding for these all
            sorted_embs.append(nonword_embeddings(300))

    final_embeddings = np.stack(sorted_embs)

    return torch.tensor(final_embeddings).float()


def match_embeddings(idx2w, w2vec_em, dim):
    embeddings = []
    voc_size = len(idx2w)
    for idx in tqdm(range(voc_size)):
        word = idx2w[idx]
        if word not in w2vec_em:
            embeddings.append(np.random.uniform(low=-1.2, high=1.2, size=(dim, )))
        else:
            embeddings.append(w2vec_em[word])

    embeddings = np.stack(embeddings)
    return embeddings


def nonword_embeddings(max_length):
    return torch.tensor(np.zeros((max_length,)))


if __name__ == '__main__':
    get_embs()
