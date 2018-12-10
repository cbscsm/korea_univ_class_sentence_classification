import os
import numpy as np

data_dir = "./amazon_reviews"
glove_dir = "./glove.6B"
embedding_dim = 300

def export_trimmed_glove_vectors():

    glove_embedding = dict()

    print("Making glove trimmed.npz, It will take few minutes...")
    with open(os.path.join(glove_dir, "glove.6B.300d.txt"), "r") as f_handle:
        for line_idx, line in enumerate(f_handle):
            line = line.strip().split(" ")
            glove_word = line[0]
            glove_embedding[glove_word] = [float(x) for x in line[1:]]

        print("# glove vocab : ", len(glove_embedding))

    with open(os.path.join(data_dir, "train.vocab"), "r") as f_handle:
        train_vocab = [line.strip() for line in list(f_handle)]
        print("# train vocab : ", len(train_vocab))

    embeddings = np.zeros([len(train_vocab) + 1, embedding_dim])

    for word_idx, word in enumerate(train_vocab):
        try:
            embeddings[word_idx] = np.asarray(glove_embedding[word.lower()])

        except KeyError:
            embeddings[word_idx] = np.random.uniform(-1, 1, 300)

    np.savez_compressed(os.path.join(glove_dir, "glove.6B.300d.trimmed.npz"), embeddings=embeddings)
    print("Saved glove vectors in numpy array : ", len(embeddings))

if __name__ == '__main__':
    export_trimmed_glove_vectors()