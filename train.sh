#!/bin/bash
#description: BERT fine-tuning

export BERT_BASE_DIR=./chinese_L-12_H-768_A-12
export DATA_DIR=./dataset
export TRAINED_CLASSIFIER=./output
export MODEL_NAME=multi_label_bert_base_epoch1

python run_multilabels_classifier.py \
  --task_name=multilabel \
  --do_train=true \
  --do_eval=true \
  --do_predict=False \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=1.0 \
  --output_dir=$TRAINED_CLASSIFIER/$MODEL_NAME