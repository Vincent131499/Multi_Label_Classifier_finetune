# -*- coding: utf-8 -*-
"""
   File Name：     generate_train_data
   Description :  生成指定训练格式的数据集
   Author :       逸轩
   date：          2019/10/19

"""

def generate_labels(classes, labels_list):
    """

    :param classes: 所有的标签列表
    :param labels_list: 每个句子对应的标签集合
    :return:
    """
    label2id_list = []
    for labels in labels_list:
        labels = labels.split(',')
        # print(labels)
        label2id = [0 for _ in range(len(classes))]
        # print(label2id)
        for i, label in enumerate(classes):
            if label in labels:
                label2id[i] = 1
        label2id_list.append(label2id)
    # print(label2id_list)
    return label2id_list

classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
labels_list = ['toxic,threat',          #[1, 0, 0, 1, 0, 0]
               'severe_toxic,obscene',
               'insult,severe_toxic',
               'identity_hate,insult,obscene'] #[0, 0, 1, 0, 1, 1]
label2id_list = generate_labels(classes, labels_list)
print(label2id_list)
