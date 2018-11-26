#!/usr/bin/env python
# encoding: utf-8

from vocab import VocabBuilder
from dataloader import DataLoader
from model import RNN


filepath = "./dataset/dataset.csv"
vocab_obj =  VocabBuilder(filepath=filepath)

word_to_index = vocab_obj.word_to_index
label_to_index = vocab_obj.label_to_index

loader = DataLoader(filepath=filepath, word_to_index=word_to_index,
                    label_to_index=label_to_index)


