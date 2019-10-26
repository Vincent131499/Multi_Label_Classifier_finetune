# Multi_Label_Classifier_finetune
微调预训练语言模型，解决多标签分类任务
该项目的目录为：
* 数据集描述
* 模型训练
* 预测
* 导出模型用于Tensorflow Serving

## 数据集描述
本文所使用的的多标签数据集来自于kaggle比赛([toxic-comment-classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge))<br>
具体示例如下：<br>
![数据描述](https://github.com/Vincent131499/Multi_Label_Classifier_finetune/raw/master/imgs/dataset_show.jpg)
<br>
其中第一个文本对应的标签为[0 0 0 0 0 0 ]
