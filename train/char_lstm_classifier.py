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

        #char, id
        self.char2id = self.data_process.char2id
        self.id2char = self.data_process.id2char

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
        char_vocab_size = len(self.id2char)

        char_embeddings = tf.get_variable(
            name="char_embeddings",
            shape=[char_vocab_size, self.hparams.char_embedding_dim],
            initializer=tf.initializers.variance_scaling(
                scale=2.0, mode="fan_in", distribution="uniform"
            )
        )

        char_embedded = tf.nn.embedding_lookup(char_embeddings, char_inputs)

        # default shape [batch_size, sentence_time, word_time, char_embedding(50)]
        # reshape [batch_size * sentence_time, word_time, char_embeddings(50)]
        char_embedded = tf.reshape(char_embedded, [-1, tf.shape(char_inputs)[-1], self.hparams.char_embedding_dim])
        # reshape [batch_size * word_time]
        char_lengths = tf.reshape(char_lengths, [-1])

        with tf.variable_scope("char-bi-RNN"):
            char_rnn_cell_forward = rnn.GRUCell(self.hparams.char_embedding_dim)
            char_rnn_cell_backward = rnn.GRUCell(self.hparams.char_embedding_dim)

            if self.hparams.dropout_keep_prob < 1.0:
                char_rnn_cell_forward = rnn.DropoutWrapper(char_rnn_cell_forward,
                                                           output_keep_prob=self._dropout_keep_prob_ph)
                char_rnn_cell_backward = rnn.DropoutWrapper(char_rnn_cell_backward,
                                                            output_keep_prob=self._dropout_keep_prob_ph)

            _, (char_output_fw_states, char_output_bw_states) = \
                tf.nn.bidirectional_dynamic_rnn(
                    char_rnn_cell_forward, char_rnn_cell_backward,
                    inputs=char_embedded,
                    sequence_length=char_lengths,
                    dtype=tf.float32
                )
            char_hiddens = tf.concat([char_output_fw_states, char_output_bw_states], axis=-1)
            char_hiddens = tf.reshape(char_hiddens,
                                  [tf.shape(char_inputs)[0], tf.shape(char_inputs)[1], self.hparams.char_embedding_dim*2])

        word_embeddings = tf.Variable(
            self.word_embeddings,
            name="word_embeddings",
            dtype=tf.float32,
            trainable=True
        )

        ## shape = [batch_size, time, embed_dim]
        word_embedded = tf.nn.embedding_lookup(word_embeddings, inputs)
        char_word_inputs = tf.concat([word_embedded, char_hiddens], axis=-1)

        with tf.variable_scope("bi-RNN"):
            # Build RNN layers
            rnn_cell_forward = rnn.GRUCell(self.hparams.rnn_hidden_dim)
            rnn_cell_backward = rnn.GRUCell(self.hparams.rnn_hidden_dim)

            # Apply dropout to RNN
            if self.hparams.dropout_keep_prob < 1.0:
                rnn_cell_forward = tf.contrib.rnn.DropoutWrapper(rnn_cell_forward,
                                                                 output_keep_prob=self._dropout_keep_prob_ph)
                rnn_cell_backward = tf.contrib.rnn.DropoutWrapper(rnn_cell_backward,
                                                                  output_keep_prob=self._dropout_keep_prob_ph)

            _, (states_fw_final, states_bw_final) = \
                tf.nn.bidirectional_dynamic_rnn(
                rnn_cell_forward, rnn_cell_backward,
                inputs=char_word_inputs,
                sequence_length=lengths,
                dtype=tf.float32
                )

            # shape = [batch_size, rnn_hidden_dim * 2]
            final_hiddens = tf.concat([states_fw_final, states_bw_final], axis=-1)

        with tf.variable_scope("layer_out"):
            layer_out = tf.layers.dense(
                inputs=final_hiddens,
                units=output_dim,
                activation=None,
                kernel_initializer=tf.initializers.variance_scaling(
                    scale=2.0, mode="fan_in", distribution="normal"
                )
            )

        return layer_out

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
                save_path = saver.save(sess, "%s/model.ckpt" % self.hparams.model,
                                       global_step=global_step_val)
                self._logger.info("Model saved at: %s" % save_path)