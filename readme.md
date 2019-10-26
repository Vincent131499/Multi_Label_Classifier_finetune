this project is based on bert. Two features were added.

1. multi-labels classification task
2. export bert model for online serving


## multi-labels

[toxic-comment-classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) data is used 
when test multi-labels model's performance. 

performance on eval dataset(p.s. i don't adjust parameter)

|class|auc|
|:---:|:---:|
|toxic|0.9832314633351247|
|severe_toxic|0.991871062741562|
|obscene|0.9903738390113118|
|threat|0.969463869224999|
|insult|0.9863569234698775|
|identity_hate|0.9872562351925845|


### data format

csv format, and it should be like follows. (same with toxic-comment-classification dataset)

```angular2html
"id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"
"0000997932d777bf","Explanation
Why the edits made under my username Hardcore Metallica Fan were reverted? They weren't vandalisms, just closure on some GAs after I voted at New York Dolls FAC. And please don't remove the template from the talk page since I'm retired now.89.205.38.27",0,0,0,0,0,0
```

header is not necessary in train file. But a classes.txt file needed to tell model how many labes will be used.




### train

in tran phase, we need to change model structure after output layer.
 
multi-classes classifier put softmax layer after output layer.

for multi-labels, softmax change to sigmoid layer, and loss change to sigmoid_cross_entropy_with_logits,
you can find it in create_model() function

### eval

eval metric change to auc of every class, tf.metrics.auc used in metric_fn()


## model export 

original bert project only save ckpt model file, but not pb file.

if you want to serving online, you need pb file. 

this problem is solved in bert [issue 146](https://github.com/google-research/bert/issues/146), 
but i can't export model after do that. So i write model_exporter to export bert model.
 