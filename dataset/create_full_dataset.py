from nltk.corpus import wordnet as wn
import random
import pandas as pd


# 全種類のlemmaのセットを作成
lemmaSet = set()
for s in wn.all_synsets(pos='n'):
    for l in s.lemma_names():
        lemmaSet.add(l)

print('num of lemma :', len(lemmaSet))
pd.to_pickle(lemmaSet, 'lemmaSet_full.pkl')



# データセットの作成
lemmaPairs = []
n_unrelated = 500000

## 同義語のペアの追加
for s in wn.all_synsets(pos='n'):
    for a in s.lemma_names():
        for b in s.lemma_names():
            lemmaPairs.append((a, b, 0))

## 上位下位、下位上位のペアの追加
for s in wn.all_synsets(pos='n'):
    hypos = s.hyponyms()
    for h in hypos:
        for a in s.lemma_names():
            for b in h.lemma_names():
                lemmaPairs.append((a, b, 1))
                lemmaPairs.append((b, a, 2))

## 無関係ペアの追加
lemmas = list(lemmaSet)
related = set((a, b) for a, b, l in lemmaPairs)
unrelated = []
for s in related:
    while len(unrelated) < n_unrelated:
        a = random.choice(lemmas)
        b = random.choice(lemmas)
        if (a, b) not in related: unrelated.append((a, b, 3))
lemmaPairs += unrelated

## 保存
print('num of lemma pairs :', len(lemmaPairs))
pd.to_pickle(lemmaPairs, 'wordnet_full.pkl')
