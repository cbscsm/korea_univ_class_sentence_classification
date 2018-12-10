import os
import collections
import numpy as np

class Data():
    def __init__(self, hparams, data_path):
        self.hparams = hparams
        self.data_path = data_path
        self.max_sentence_length = 0
        self.max_word_length = 0

        #word_vocab -> train.vocab
        self.get_vocab()
        #char_vocab
        self.get_char_vocab()

    def get_vocab(self):

        #train_vocab
        with open(os.path.join(self.data_path,"train.vocab"),"r") as f_handle:
            self.id2word = [line.strip() for line in list(f_handle) if len(line.strip()) > 0]

        self.word2id = dict()
        for i, word in enumerate(self.id2word):
            self.word2id[word] = i

        #label.vocab
        with open(os.path.join(self.data_path,"label.vocab"),"r") as f_handle:
            labels = [l.strip() for l in list(f_handle) if len(l.strip()) > 0]
        self.id2label = labels
        self.label2id = dict()

        for i, label in enumerate(labels):
            self.label2id[label] = i

    def get_char_vocab(self):
        self.id2char = list()

        with open(os.path.join(self.data_path,"train.inputs"),"r") as f_handle:
            text = [l.strip() for l in list(f_handle) if len(l.strip()) > 0]
            full_text = ""
            for sentence in text:
                full_text += "".join(sentence.split(" "))

        alphabet_counter = collections.Counter(full_text).most_common()
        for alphabet, count in alphabet_counter:
            self.id2char.append(alphabet)

        self.char2id = dict()
        self.id2char.insert(0, "<PAD>")

        for i, char in enumerate(self.id2char):
            self.char2id[char] = i

    def load_data(self, data_type="train"):
        inputs, labels, lengths = [], [], []

        char_inputs, char_inputs_temp = [], []
        char_lengths, char_lengths_temp = [], []

        with open(os.path.join(self.data_path,"%s.inputs" % data_type),"r") as f_handle:
            for i, sentence in enumerate(list(f_handle)):

                inputs.append(sentence.strip().split(' '))
                sentence_len = len(sentence.strip().split(' '))

                if len(sentence.strip().split(' ')) < self.max_sentence_length:
                    self.max_sentence_length = sentence_len

                #make the list about char lengths
                for words in sentence.strip().split(' '):
                    char_inputs_temp.append(list(words))
                    char_lengths_temp.append(len(list(words)))

                    if len(list(words)) > self.max_word_length:
                        self.max_word_length = len(list(words))

                char_inputs.append(char_inputs_temp)
                char_lengths.append(char_lengths_temp)
                char_inputs_temp = []
                char_lengths_temp = []

        with open(os.path.join(self.data_path, "%s.labels" % data_type), "r") as f_handle:
            for i, sentence in enumerate(list(f_handle)):
                labels.append(sentence.strip().split(' '))

        for sentence in inputs:
            lengths.append(len(sentence))

        return (char_inputs, char_lengths), (inputs, labels, lengths)

    def data_id(self, inputs, labels, chars):
        inputs_id = inputs
        labels_id = labels
        chars_id = chars

        for sentence in inputs_id:
            for i, word in enumerate(sentence):
                try:
                    sentence[i] = self.word2id[word]

                except KeyError:
                    sentence[i] = len(self.word2id)

        for sentence in labels_id:
            for i, label in enumerate(sentence):
                sentence[i] = self.label2id[label]

        for sentence in chars_id:
            for i, word in enumerate(sentence):
                for j, char in enumerate(word):
                    try:
                        sentence[i][j] = self.char2id[char]
                    except KeyError:
                        print("char key error : ", char)
                        self.char2id[char] = len(self.id2char)
                        sentence[i][j] = self.char2id[char]

        return inputs_id, labels_id, chars_id

    def get_batch_data(self, input_id, labels_id, train_lengths, chars_id, char_lengths, iter, batch_size):
        idx = iter * batch_size

        batch_inputs = input_id[idx:idx + batch_size]
        batch_labels = labels_id[idx:idx + batch_size]
        batch_lengths = train_lengths[idx:idx + batch_size]

        batch_char_inputs = chars_id[idx:idx + batch_size]
        batch_char_lengths = char_lengths[idx:idx + batch_size]

        max_sentence_len = max(batch_lengths)

        max_word_length = 0
        for char_len_sentence in batch_char_lengths:
            if max_word_length < max(char_len_sentence):
                max_word_length = max(char_len_sentence)

        #sentence padding
        for sentence in batch_inputs:
            if len(sentence) < max_sentence_len:
                sentence.extend([0]*(max_sentence_len-len(sentence)))

        #batch_char_inputs: padding
        for words_list in batch_char_inputs:
            if len(words_list) < max_sentence_len:
                for i in range(max_sentence_len - len(words_list)):
                    words_list.append([0])

            for word in words_list:
                if len(word) < max_word_length:
                    word.extend([0]*(max_word_length - len(word)))

        #batch_char_lengths: padding
        for words_length in batch_char_lengths:
            if len(words_length) < max_sentence_len:
                for i in range(max_sentence_len - len(words_length)):
                    words_length.append(0)

        batch_labels_temp = list()
        for sentence in batch_labels:
            batch_labels_temp.extend(sentence)

        batch_labels = batch_labels_temp

        return batch_inputs, batch_labels, batch_lengths, batch_char_inputs, batch_char_lengths
