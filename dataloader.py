#!/usr/bin/env python
# encoding: utf-8

import torch
import pandas as pd
import numpy as np
import utils


class DataLoader(object):
    def __init__(self, filepath, word_to_index, label_to_index, batch_size=32):
        self.filepath = filepath
        self.batch_size = batch_size
        self.word_to_index = word_to_index
        self.label_to_index = label_to_index

        self._parse_data()


        self.n_samples = len(self.samples)
        self.n_batches = int(self.n_samples / self.batch_size)

        self.max_length = self._get_max_length()
        self._shuffle_indices()

        self.report()

    def _parse_data(self):
        def generate_body_index(body_text):
            indices = []
            for word in body_text:
                if word in self.word_to_index:
                    indices.append(self.word_to_index[word])
                else:
                    indices.append(self.word_to_index["__UNK__"])
            return indices

        def generate_label_index(label):
            return self.label_to_index[label]

        df = pd.read_csv(self.filepath, delimiter="\t", names=["body", "label"])

        df["body"] = df["body"].apply(utils._tokenize)
        df['body'] = df['body'].apply(lambda x: generate_body_index(x))
        df["label"] = df["label"].apply(lambda x: generate_label_index(x))

        self.samples = df.values.tolist()


    def _shuffle_indices(self):
        self.indices = np.random.permutation(self.n_samples)
        self.index = 0
        self.batch_index = 0


    def _get_max_length(self):
        length = 0
        for sample in self.samples:
            # print(sample)
            length = max(length, len(sample[0]))
        return length


    def _padding(batch_x):
        batch_s = sorted(batch_x, key=lambda x: len(x))
        # longest one
        size = len(batch_s[-1])
        for i, x in enumerate(batch_x):
            missing = size - len(x)
            batch_x[i] = batch_x[i] + [0 for _ in range(missing)]
        return batch_x


    def _create_batch(self):
        batch = []
        n = 0
        while n < self.batch_size:
            _index = self.indices[self.index]
            batch.append(self.samples[_index])
            self.index += 1
            n += 1
        self.batch_index += 1

        string, label = tuple(zip(*batch))

        seq_lengths = torch.LongTensor(list(map(len, string)))

        seq_tensor = torch.zeros((len(string), seq_lengths.max()), dtype=torch.long)

        for idx, (seq, seqlen) in enumerate(zip(string, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

        # sort by length
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]

        label = torch.LongTensor(label)
        label = label[perm_idx]

        return seq_tensor, label, seq_lengths


    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self._shuffle_indices()
        for i in range(self.n_batches):
            if self.batch_size == self.n_batches:
                raise StopIteration()
            yield self._create_batch()

    def show_samples(self, n=10):
        for sample in self.samples[:n]:
            print(sample)

    def report(self):
        print('| Data Loader info: ')
        print('| # samples: {}'.format(len(self.samples)))
        print('| max len: {}'.format(self.max_length))
        print('| vocab: {}'.format(len(self.word_to_index)))
        print('| batches: {} (batch_size = {})'.format(self.n_batches, self.batch_size))
