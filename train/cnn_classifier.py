import os
import logging
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

from data_process import Data

class TextClassifier:
    def __init__(self, hparams, data_dir):
        self.hparams = hparams
        self.data_dir = data_dir
        #logger
        self._logger = logging.getLogger(__name__)

        #data_process
        self.data_process = Data(self.hparams, self.data_dir)
        (self.char_inputs, self.char_lengths), (self.inputs, self.labels, self.lengths) = \
            self.data_process.load_data()

        # word, id
        self.word2id = self.data_process.word2id  # dict()
        self.id2word = self.data_process.id2word  # vocabulary

        # label, id
        self.label2id = self.data_process.label2id
        self.id2label = self.data_process.id2label

        # pre-trained word2vec
        with np.load(os.path.join(self.hparams.glove_dir, "glove.6B.300d.trimmed.npz")) as pretrained_data:
            self.word_embeddings = pretrained_data["embeddings"]
            print(np.shape(self.word_embeddings))

    def _inference(self, inputs: tf.Tensor, lengths: tf.Tensor, char_inputs: tf.Tensor, char_lengths: tf.Tensor):
        print("Building graph for model: Text Classifier")

        # Number of possible output cateIories.
        output_dim = len(self.id2label) # output_dim -> 2

        word_embeddings = tf.Variable(
            self.word_embeddings,
            name="word_embeddings",
            dtype=tf.float32,
            trainable=True
        )

        ## shape = [batch_size, time, embed_dim]
        word_embedded = tf.nn.embedding_lookup(word_embeddings, inputs)
        word_feature_map = tf.expand_dims(word_embedded, -1)
        # for i, filter_size in enumerate(self.hparams.filter_size):
        #     with tf.variable_scope("CNN-%s" % filter_size):
        #

        # Convolution & Maxpool
        features = []
        for size in self.hparams.filter_size:
            with tf.variable_scope("CNN_filter_%d" % size):
                # Add padding to mark the beginning and end of words.
                pad_height = size - 1
                pad_shape = [[0, 0], [pad_height, pad_height], [0, 0], [0, 0]]
                word_feature_map = tf.pad(word_feature_map, pad_shape)
                feature = tf.layers.conv2d(
                    inputs=word_feature_map,
                    filters=self.hparams.num_filters,
                    kernel_size=[size, self.hparams.embedding_dim],
                    use_bias=False
                )
                # shape = [batch, time, 1, out_channels]
                feature = tf.reduce_max(feature, axis=1)
                feature = tf.squeeze(feature)
                feature = tf.reshape(feature, [tf.shape(inputs)[0], self.hparams.num_filters])
                # shape = [batch, out_channels]
                features.append(feature)

        # shape = [batch, out_channels * num_filters]
        layer_out = tf.concat(features, axis=1)

        with tf.variable_scope("layer_out"):
            logits = tf.layers.dense(
                inputs=layer_out,
                units=output_dim,
                activation=None,
                kernel_initializer=tf.initializers.variance_scaling(
                    scale=2.0, mode="fan_in", distribution="normal"
                )
            )

        return logits

    def make_placeholder(self):

        self.inputs_ph = tf.placeholder(tf.int32, shape=[None, None], name="train_input_ph")
        self.labels_ph = tf.placeholder(tf.int32, shape=[None], name="train_label_ph")
        self.lengths_ph = tf.placeholder(tf.int32, shape=[None], name="train_lengths_ph")

        #[batch_size, word_time, char_time]
        self.char_inputs_ph = tf.placeholder(tf.int32, shape=[None, None, None], name="char_input_ph")
        self.char_lengths_ph = tf.placeholder(tf.int32, shape=[None, None], name="char_lengths_ph")

        self._dropout_keep_prob_ph = tf.placeholder(tf.float32, shape=[], name="dropout_keep_prob")

    def make_feed_dict(self, batch_data):
        feed_dict = {}
        batch_inputs, batch_labels, batch_lengths, batch_char_inputs, batch_char_lengths = batch_data

        # word-level
        feed_dict[self.inputs_ph] = batch_inputs
        feed_dict[self.labels_ph] = batch_labels
        feed_dict[self.lengths_ph] = batch_lengths

        # char-level
        feed_dict[self.char_inputs_ph] = batch_char_inputs
        feed_dict[self.char_lengths_ph] = batch_char_lengths
        feed_dict[self._dropout_keep_prob_ph] = self.hparams.dropout_keep_prob

        return feed_dict

    def build_graph(self):

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # logits
        with tf.variable_scope("inference", reuse=False):
            logits = self._inference(self.inputs_ph, self.lengths_ph, self.char_inputs_ph, self.char_lengths_ph)

        with tf.name_scope("cross_entropy"):
            loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels_ph,
                                                                     name="cross_entropy")
            self.loss_op = tf.reduce_mean(loss_op, name='cross_entropy_mean')
            self.train_op = tf.train.AdamOptimizer().minimize(loss_op, global_step=self.global_step)

        eval = tf.nn.in_top_k(logits, self.labels_ph, 1)
        correct_count = tf.reduce_sum(tf.cast(eval, tf.int32))
        with tf.name_scope("accuracy"):
            self.accuracy = tf.divide(correct_count, tf.shape(self.labels_ph)[0])

    def train(self):
        sess = tf.Session()

        with sess.as_default():
            global_step = tf.Variable(0, name='global_step', trainable=False)

            # build placeholder
            self.make_placeholder()
            # build train graph
            self.build_graph()

            # checkpoint file saver
            saver = tf.train.Saver()

            # get data
            inputs_id, labels_id, chars_id = \
                self.data_process.data_id(self.inputs, self.labels, self.char_inputs)

            total_batch = int(len(inputs_id) / self.hparams.batch_size) + 1
            tf.global_variables_initializer().run()
            for epochs_completed in range(self.hparams.num_epochs):

                for iter in range(total_batch):
                    batch_data = self.data_process.get_batch_data(inputs_id, labels_id, self.lengths,
                                                                  chars_id, self.char_lengths,
                                                                  iter, self.hparams.batch_size)

                    accuracy_val, loss_val, global_step_val, _ = sess.run(
                        [self.accuracy, self.loss_op, self.global_step, self.train_op],
                        feed_dict=self.make_feed_dict(batch_data)
                    )

                    if global_step_val % 10 == 0:
                        self._logger.info("[Step %d] loss: %.4f, accuracy: %.2f%%" % (
                            global_step_val, loss_val, accuracy_val * 100))

                self._logger.info("End of epoch %d." % (epochs_completed + 1))
                save_path = saver.save(sess, os.path.join(self.hparams.root_dir,
                                                          "saves_%s/model.ckpt" % (self.hparams.model)),
                                       global_step=global_step_val)
                self._logger.info("Model saved at: %s" % save_path)