import numpy  as np
import pandas as pd
from nltk.corpus import wordnet as wn


# 'multi_words_term' -> [np(300), np(300), np(300)]
def vectorize_lemma(w2v, lemma, max_seq_len):
    lemma = lemma.split('_')
    vecs = []
    for w in lemma:
        if w in w2v: vecs.append(w2v[w])
        else       : vecs.append(np.zeros(300))
    # padding
    n_pad = max_seq_len - len(lemma)
    vecs += [np.zeros(300)] * n_pad

    return np.array(vecs)


def main():
    #
    word2vec = pd.read_pickle('word2vec.pkl')
    lemmaSet = pd.read_pickle('../dataset/lemmaSet_filtered.pkl')
    dataset  = pd.read_pickle('../dataset/wordnet_filtered.pkl')

    max_seq_len = max(max(len(a.split('_')), len(b.split('_'))) for a, b, l in dataset)
    # vec_size = vectorizer['example'][0].shape[0]

    vectorizer_w2v = {l: vectorize_lemma(word2vec, l, max_seq_len) for l in lemmaSet}

    print('vocab size of vectorizer :', len(vectorizer_w2v))
    print('maximum sequence length  :', max(len(l.split('_')) for l in vectorizer_w2v.keys()))

    pd.to_pickle(vectorizer_w2v, 'vectorizer_w2v.pkl')


if __name__ == '__main__': main()
