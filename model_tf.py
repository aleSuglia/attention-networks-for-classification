import numpy as np
import tensorflow as tf


class HierarchicalAttentionNetwork:
    def __init__(self,
                 documents_input,
                 num_sentences_input,
                 max_sentence_len_input,
                 batch_size_input,
                 params
                 ):
        self.params = params
        self._documents_input = documents_input
        self._num_sentences_input = num_sentences_input
        self._max_sentence_len_input = max_sentence_len_input
        self._batch_size_input = batch_size_input

        with tf.variable_scope("han"):
            embeddings_matrix = tf.get_variable(
                "embeddings_matrix",
                shape=(params["num_tokens"], params["embedding_size"]),
                initializer=tf.random_normal_initializer()
            )
            documents_embeddings = tf.nn.embedding_lookup(embeddings_matrix, documents_input)

            bigru_word_docs = self._rnn_word_encoder(documents_embeddings)
            bigru_word_attention_docs = self._rnn_word_attention(bigru_word_docs)
            self._docs_representations = self._rnn_sent_attention(self._rnn_sent_encoder(bigru_word_attention_docs))
            self._output = self._output_layer(self._docs_representations)

    def _rnn_word_encoder(self, documents_embeddings):
        with tf.variable_scope("rnn_word_encoder"):
            documents_embeddings = tf.reshape(documents_embeddings, (self._batch_size_input * self._num_sentences_input,
                                                                     self._max_sentence_len_input,
                                                                     self.params["embedding_size"]))
            reshaped_documents_input = tf.reshape(self._documents_input, (self._batch_size_input *
                                                                          self._num_sentences_input,
                                                                          self._max_sentence_len_input))
            sequence_length = tf.reduce_sum(
                tf.cast(tf.not_equal(reshaped_documents_input, tf.zeros_like(reshaped_documents_input)), tf.int32),
                1
            )
            bigru_word_docs, _ = tf.nn.bidirectional_dynamic_rnn(
                tf.contrib.rnn.GRUCell(self.params["gru_output_size"]),
                tf.contrib.rnn.GRUCell(self.params["gru_output_size"]),
                documents_embeddings,
                dtype=tf.float32,
                sequence_length=sequence_length
            )

        return tf.reshape(
            tf.concat(bigru_word_docs, 2),
            (
                self._batch_size_input,
                self._num_sentences_input,
                self._max_sentence_len_input,
                2 * self.params["gru_output_size"]
            )
        )

    def _rnn_word_attention(self, bigru_word_docs):
        with tf.variable_scope("rnn_word_attention"):
            weights = tf.expand_dims(tf.nn.softmax(
                tf.squeeze(
                    tf.contrib.layers.fully_connected(
                        bigru_word_docs,
                        num_outputs=1,
                        activation_fn=tf.tanh,
                        biases_initializer=tf.zeros_initializer(),
                        weights_initializer=tf.random_normal_initializer()
                    )
                )
            ), 3)

        return tf.reduce_sum(tf.multiply(weights, bigru_word_docs), 2)

    def _rnn_sent_encoder(self, docs_sent_attention):
        with tf.variable_scope("sent_encoder"):
            valid_words = tf.reduce_sum(
                self._documents_input,
                2
            )
            sequence_length = tf.reduce_sum(
                tf.cast(tf.not_equal(valid_words, tf.zeros_like(valid_words)), tf.int32),
                1
            )
            docs_sent_attention.set_shape((None, None, 2 * self.params["gru_output_size"]))
            bigru_sent_docs, _ = tf.nn.bidirectional_dynamic_rnn(
                tf.contrib.rnn.GRUCell(2 * self.params["gru_output_size"]),
                tf.contrib.rnn.GRUCell(2 * self.params["gru_output_size"]),
                docs_sent_attention,
                dtype=tf.float32,
                sequence_length=sequence_length
            )

        return tf.concat(bigru_sent_docs, 2)

    def _rnn_sent_attention(self, bigru_sent_docs):
        with tf.variable_scope("rnn_sent_attention"):
            weights = tf.expand_dims(tf.nn.softmax(
                tf.squeeze(
                    tf.contrib.layers.fully_connected(
                        bigru_sent_docs,
                        num_outputs=1,
                        activation_fn=tf.tanh,
                        biases_initializer=tf.random_normal_initializer(),
                        weights_initializer=tf.zeros_initializer()
                    )
                )
            ), 2)

        return tf.reduce_sum(tf.multiply(weights, bigru_sent_docs), 1)

    def _output_layer(self, docs_representations):
        docs_representations.set_shape((None, 4 * self.params["gru_output_size"]))

        with tf.variable_scope("output_layer"):
            output = tf.contrib.layers.fully_connected(
                docs_representations,
                self.params["num_classes"],
                weights_initializer=tf.random_normal_initializer(),
                biases_initializer=tf.zeros_initializer()
            )

        return output

    def loss(self, targets_input):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self._output,
            labels=tf.one_hot(targets_input, self.params["num_classes"], axis=1)
        ))

    @property
    def output(self):
        return self._output


def main():
    data = [
        [
            [1, 1, 0, 0],
            [2, 3, 4, 0]
        ],
        [
            [1, 2, 3, 0],
            [2, 2, 3, 0]
        ],
        [
            [1, 10, 2, 0],
            [3, 4, 5, 0]
        ]
    ]

    batch_size = 3
    max_num_sentences = 2
    max_sentence_len = 4
    num_tokens = 11
    embedding_size = 10
    gru_output_size = 20
    output_layer_size = 10
    params = {
        "num_tokens": num_tokens,
        "embedding_size": embedding_size,
        "gru_output_size": gru_output_size,
        "output_layer_size": output_layer_size
    }

    session = tf.InteractiveSession()

    num_sentences_input = tf.placeholder(dtype=tf.int32)
    batch_size_input = tf.placeholder(dtype=tf.int32)
    sentence_len_input = tf.placeholder(dtype=tf.int32)
    documents_input = tf.placeholder(dtype=tf.int32, shape=(None, None, max_sentence_len))
    targets_input = tf.placeholder(dtype=tf.int32, shape=(None,))

    han_model = HierarchicalAttentionNetwork(documents_input,
                                             num_sentences_input,
                                             sentence_len_input,
                                             batch_size_input,
                                             params)

    # loss_function = han_model.loss(targets_input)
    #
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    # train_op = optimizer.minimize(loss_function)
    #
    session.run(tf.global_variables_initializer())

    res = session.run(han_model._docs_representations,
                      {
                          documents_input: data,
                          batch_size_input: batch_size,
                          num_sentences_input: max_num_sentences,
                          sentence_len_input: max_sentence_len,
                          targets_input: np.random.randint(0, 10, (batch_size,))
                      }
                      )

    print(res.shape)


if __name__ == "__main__":
    main()
