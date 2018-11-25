#!/usr/bin/env python
# encoding: utf-8

import wget
import os
import pandas as pd
import glob
import jieba

def download():
    print("downloading...")
    for f in glob.glob("*.tmp"):
        os.remove(f)
        print("cleaned tmp files")

    training_dataset_url = "http://tcci.ccf.org.cn/conference/2018/dldoc/trainingdata06.rar"
    wget.download(training_dataset_url, "dataset.rar")


def clean():

    data = pd.read_csv("sample.csv")
    data = data[["question_title", "tag_ids"]]
    # data = data[:10]
    a = data.values.tolist()

    tags = pd.read_csv("topic_2_id.csv")
    tags = tags[["topic_name", "topic_id"]]
    tags_dict = dict(zip(tags.topic_id, tags.topic_name))

    with open("dataset.csv", "w") as f:
        for sample in a:
            sample_content = sample[0]
            sample_tags = sample[1].split("|")
            for tag_id in sample_tags:
                tag_name = tags_dict[int(tag_id)]
                f.write(" ".join(jieba.lcut(sample_content.replace(" ", "")))+"\t"+tag_name+"\n")


if __name__ == "__main__":
    # download()
    clean()
