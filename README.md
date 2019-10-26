# Multi_Label_Classifier_finetune
微调预训练语言模型，解决多标签分类任务
<br>
该项目的目录为：
* 数据集描述
* 模型训练
* 预测
* 导出模型用于Tensorflow Serving

## 数据集描述
本文所使用的的多标签数据集来自于kaggle比赛([toxic-comment-classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge))<br>
具体示例如下：<br>
![数据描述](https://github.com/Vincent131499/Multi_Label_Classifier_finetune/raw/master/imgs/data_show.jpg)
<br>
标签描述：<br>
![标签描述](https://github.com/Vincent131499/Multi_Label_Classifier_finetune/raw/master/imgs/labels_show.jpg)
<br>
上面有2句示例，第一行分别对应(id,text,labels)，其中labels通过类似于one-hot的方式进行了转换，这里就变成了'1,1,1,0,1,0'，比对标签文件中标签的顺序，表示该文本对应的标签为'toxic,severe_toxic,obscene,insult'<br>

## 模型训练
运行命令：
```Bash
bash train.sh
```
训练命令的参数说明：<br>
* BERT_BASE_DIR:预训练语言模型所在路径
* DATA_DIR:训练集所在路径
* TRAINED_CLASSIFIER:训练的模型所在目录
* MODEL_NAME：训练的模型名称
除此之外，还要根据自己的数据集和显存情况指定max_seq_length、train_batch_size和num_train_epochs。<br>
训练后模型的评估效果如下所示：
![模型评估效果](https://github.com/Vincent131499/Multi_Label_Classifier_finetune/raw/master/imgs/model_perform.jpg)

注意：<br>
在训练阶段：
我们必须修改在output layer后的模型架构。多类分类器将softmax层放置在输出层之后。<br>
对于多标签，softmax更改为Sigmoid层，loss更改为sigmoid_cross_entropy_with_logits，可以在create_model（）函数中找到它。<br>
在评估阶段：
评估标准通过使用的tf.metrics.auc修改为每个类别的auc。具体的可以在metric_fn（）中看到.<br>

## 预测
运行命令：
```Bash
python run_classifier_predict_online.py
```
注意：<br>
在运行前需要在文件中指定模型路径，修改BERT_BASE_DIR参数，该参数的值为你训练好的模型所在路径（还需要将预训练语言模型中的bert_config.json和vocab.txt两个文件复制到你训练好的模型目录下面，因为模型预测时需要加载这两个文件）<br>
运行效果如下所示：<br>
![预测效果](https://github.com/Vincent131499/Multi_Label_Classifier_finetune/raw/master/imgs/predict_show.jpg)

## 导出模型用于Tensorflow Serving
由于通过Tensorflow Seving部署模型需要用到pb格式的文件，故在这里也提供了模型转换的功能。<br>
运行命令：
```Bash
python model_exporter.py
```
注意：<br>
在运行前需要修改一下data_path、labels_num和export_path这三个参数的值。其中data_path为你训练好的模型所在路径，labels_num为标签的数目，export_path为导出后的pd模型存储的位置。
导出后如下所示：<br>
![导出模型示例](https://github.com/Vincent131499/Multi_Label_Classifier_finetune/raw/master/imgs/exported_show.jpg)
