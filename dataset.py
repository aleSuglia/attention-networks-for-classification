import multiprocessing

import numpy as np
from sklearn.datasets import fetch_20newsgroups


# Reference: https://github.com/fchollet/keras/blob/master/keras/preprocessing/sequence.py
def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads each sequence to the same length (length of the longest sequence).
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.
    Supports post-padding and pre-padding (default).
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    # Raises
        ValueError: in case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def create_batch_data(data, indexes):
    batch_data = []
    max_sentence_len = 0
    max_num_sentences = 0

    for i in indexes:
        num_sentences = len(data[i])
        max_num_sentences = max(max_num_sentences, num_sentences)
        for j in range(num_sentences):
            max_sentence_len = max(max_sentence_len, len(data[i][j]))
        batch_data.append([token for token in data[i]])

    for i in range(len(batch_data)):
        batch_data[i] = np.pad(
            pad_sequences(batch_data[i], maxlen=max_sentence_len, padding="post"),
            pad_width=[(0, max_num_sentences - len(batch_data[i])), (0, 0)],
            mode="constant"
        )

    return np.array(batch_data), max_num_sentences, max_sentence_len


def process_20newsgroup_dataset(nlp):
    train_dataset = fetch_20newsgroups(subset="train")
    test_dataset = fetch_20newsgroups(subset="test")
    processed_train = []
    processed_test = []

    print("-- Processing training dataset")
    for doc in nlp.pipe(train_dataset.data, batch_size=10000, n_threads=multiprocessing.cpu_count()):
        processed_train.append([[token.orth for token in sent if not token.is_stop] for sent in doc.sents])

    print("-- Processing test dataset")
    for doc in nlp.pipe(test_dataset.data, batch_size=10000, n_threads=multiprocessing.cpu_count()):
        processed_test.append([[token.orth for token in sent if not token.is_stop] for sent in doc.sents])

    train_dataset.data = processed_train
    test_dataset.data = processed_test

    return train_dataset, test_dataset
