# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import sys
import tensorflow as tf

import modeling

# csv.field_size_limit(sys.maxsize)

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("data_path", 'output/accu_multi_label_bert_base_epoch1', "saved model path")

flags.DEFINE_string("labels_num", '6', "number of your labels")

flags.DEFINE_string("export_path", 'exported', "savedModel export path")


class ModelTransfer(object):
    def __init__(self, max_seq_length=128):
        self.max_seq_length = max_seq_length
        self.labels_num = int(FLAGS.labels_num)
        self.bert_config_file = os.path.join(FLAGS.data_path, 'bert_config.json')

    def _create_model(self, bert_config, is_training, input_ids, input_mask, segment_ids,
                      labels, num_labels, use_one_hot_embeddings):
        """Creates a classification model."""
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        # In the demo, we are doing a simple classification task on the entire
        # segment.
        #
        # If you want to use the token-level output, use model.get_sequence_output()
        # instead.
        output_layer = model.get_pooled_output()

        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            if is_training:
                # I.e., 0.1 dropout
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)

            return (loss, per_example_loss, logits, probabilities)

    def transfer(self):
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        sess = tf.Session(config=gpu_config)
        print("going to restore checkpoint")
        bert_config = modeling.BertConfig.from_json_file(self.bert_config_file)

        input_ids = tf.placeholder(tf.int32, [1, self.max_seq_length], name="input_ids")
        input_mask = tf.placeholder(tf.int32, [1, self.max_seq_length], name="input_mask")
        segment_ids = tf.placeholder(tf.int32, [1, self.max_seq_length], name="segment_ids")
        # multi task classication problem need to modify this
        label_ids = tf.placeholder(tf.int32, [1], name="label_ids")

        total_loss, per_example_loss, logits, probabilities = self._create_model(
            bert_config, False, input_ids, input_mask, segment_ids,
            label_ids, self.labels_num, False)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.data_path))
        tf.saved_model.simple_save(sess,
                                   FLAGS.export_path,
                                   inputs={
                                       'label_ids': label_ids,
                                       'input_ids': input_ids,
                                       'input_mask': input_mask,
                                       'segment_ids': segment_ids
                                   },
                                   outputs={"probabilities": probabilities})
        print('savedModel export finished')


if __name__ == '__main__':
    # path of model file and bert_config.json file
    flags.mark_flag_as_required("data_path")
    # export model saved path
    flags.mark_flag_as_required("export_path")
    flags.mark_flag_as_required("labels_num")
    ModelTransfer().transfer()
