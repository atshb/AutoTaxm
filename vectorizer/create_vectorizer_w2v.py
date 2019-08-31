import numpy  as np
import pandas as pd





def vectorize_lemma(w2v, lemma):
    lemma = lemma.split('_')
    vecs = []
    for w in lemma:
        if w in w2v: vecs.append(w2v[w])
        else       : vecs.append(np.zeros(300))
    return vecs

def main():
    word2vec = pd.read_pickle('word2vec.pkl')
    dataset  = pd.read_pickle('../dataset/wordnet_full.pkl')

    vectorizer_w2v = dict()
    for a, b, l in dataset:
        vecs_a = vectorize_lemma(word2vec, a)
        vecs_b = vectorize_lemma(word2vec, b)
        # if (a, b) in vectorizer_w2v: print(a, b)
        vectorizer_w2v[(a, b)] = (vecs_a, vecs_b)

    print(len(vectorizer_w2v))
    pd.to_pickle(vectorizer_w2v, 'vectorizer_w2v.pkl')

if __name__ == '__main__': main()
