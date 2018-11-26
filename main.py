#!/usr/bin/env python
# encoding: utf-8

from vocab import VocabBuilder
from dataloader import DataLoader
from model import RNN


filepath = "./dataset/dataset.csv"
vocab_obj =  VocabBuilder(filepath=filepath)

word_to_index = vocab_obj.word_to_index
label_to_index = vocab_obj.label_to_index

index_to_label = {}
for label, index in label_to_index.items():
    index_to_label[index] = label

loader = DataLoader(filepath=filepath, word_to_index=word_to_index,
                    label_to_index=label_to_index, batch_size=128)

vocab_size = len(word_to_index)
embedding_size = 128
num_output = len(label_to_index)

model = RNN(vocab_size=vocab_size, embed_size=embedding_size,
            num_output=num_output, rnn_model="LSTM",
            use_last=True, hidden_size=128,
            embedding_tensor=None, num_layers=2, batch_first=True)

model.to("cuda:0")

import torch
import torch.nn as nn
import torch.optim as optim
from utils import accuracy, AverageMeter

optimizer = optim.Adam(model.parameters(), lr=0.005)

criterion = nn.CrossEntropyLoss()
clip=0.25


def train(epoches):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    for idx, (input_data, input_label, seq_lengths) in enumerate(loader):
        input_data = input_data.to("cuda:0")
        input_label = input_label.to("cuda:0")

        output = model(input_data, seq_lengths)
        loss = criterion(output, input_label)
        # print(loss.item())

        prec1 = accuracy(output.data, input_label, topk=(1,))

        losses.update(loss.item(), input_data.size(0))
        top1.update(prec1[0][0], input_data.size(0))

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if idx % 100 == 0:
            print("| Epoch [{0}][{1}/{2}] Loss {loss.val:.4f} ({loss.avg:.4f})"
                  "Prec@1 {top1.val:.4f} ({top1.avg:.4f})".format(epoches, idx, len(loader),
                                                                loss=losses, top1=top1))



def do_train():
    for i in range(50):
        train(i)
        torch.save(model.state_dict(), "./checkpoints/checkpoint_{}.pth".format(i))


def do_eval():
    model.load_state_dict(torch.load("./checkpoints/checkpoint_49.pth"))

    model.eval()

    test_sample = "如何 最快 确定 自己 乘坐 的 飞机 自 几号 航站楼 起飞 ？"
    # test_sample = "话剧 演员 说错 台词 怎么办 ？"
    test_sample_idx = []
    for _ in test_sample.split():
        try:
            test_sample_idx.append(word_to_index[_])
        except:
            test_sample_idx.append(word_to_index["__UNK__"])

    test_sample_tensor = torch.tensor([test_sample_idx], dtype=torch.long)

    test_sample_tensor  = test_sample_tensor.to("cuda:0")
    lengths = torch.tensor([len(test_sample_idx)], dtype=torch.long)

    output = model(test_sample_tensor, lengths)
    # _, predicted = torch.max(output, 5)
    _, predicted = torch.topk(output, 5)

    predicted = predicted.cpu().numpy()
    print(predicted[0])
    for _ in predicted[0]:
        print(index_to_label[_])


if __name__ == "__main__":
    # do_train()
    do_eval()
