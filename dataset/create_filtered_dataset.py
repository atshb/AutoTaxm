from nltk.corpus import wordnet as wn
import random
import pandas as pd
import collections
import itertools


# sequence lengthの分布を表示する
def print_histogram_of_seq_length(lemmaSet):
    counter = collections.Counter([len(l.split('_')) for l in lemmaSet])

    print('-' * 20)
    for c in itertools.count(1):
        if c in counter: print(f'{c:>2} : {counter[c]:>7}')
        else: break
    print('-' * 20)
    print('total :', len(lemmaSet))
    print('-' * 20)


# parameters
def main():
    max_seq_len = 5
    registered_in_w2v = False

    # load dataset file
    lemmaSet = pd.read_pickle('lemmaSet_full.pkl')
    dataset  = pd.read_pickle('wordnet_full.pkl')
    print(len(dataset))
    print_histogram_of_seq_length(lemmaSet)

    # filtering
    lemmaSet = set(l for l in lemmaSet if len(l.split('_')) <= max_seq_len)
    dataset  = [(a, b, l) for a, b, l in dataset if a in lemmaSet and b in lemmaSet]

    # save as other pickle file
    pd.to_pickle(lemmaSet, 'lemmaSet_filtered.pkl')
    pd.to_pickle(dataset , 'wordnet_filtered.pkl')
    print(len(dataset))


if __name__ == '__main__': main()
