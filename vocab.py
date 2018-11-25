#!/usr/bin/env python
# encoding: utf-8

import pandas as pd
import utils
from collections import Counter

class VocabBuilder(object):
    def __init__(self, filepath, min_sample=10):
        self.filepath = filepath
        self.word_to_index = {}
        self.label_to_index = {}
        self.min_sample=min_sample

        self.count_from_file()


    def count_from_file(self, tokenizer=utils._tokenize):
        padding_marker = "__PADDING__"
        unknown_marker = "__UNK__"
        df = pd.read_csv(self.filepath, delimiter="\t", names=["content", "label"])
        # counting labels
        label_index = list(set(df["label"].values))
        label_index.sort()

        for idx, label in enumerate(label_index):
            self.label_to_index[label] = idx

        # counting content
        df["content"] = df["content"].apply(lambda x: tokenizer(x))
        word_count = Counter([tkn for sample in df["content"].tolist() for tkn in sample])
        print("| Original vocab size: {}".format(len(word_count)))

        _word_count = list(filter(lambda x: self.min_sample<=x[1], word_count.items()))
        tokens = list(zip(*_word_count))[0]

        self.word_to_index = { tkn: i for i, tkn in enumerate([padding_marker, unknown_marker] + sorted(tokens)) }
        print('| Turncated vocab size:{} (removed:{})'.format(len(self.word_to_index), len(word_count) - len(self.word_to_index)))



if __name__ == "__main__":
    vocab_obj = VocabBuilder("./dataset/dataset.csv", min_sample=1000)
    print(vocab_obj.filepath)

    print(vocab_obj.word_to_index)
    print(vocab_obj.label_to_index)
