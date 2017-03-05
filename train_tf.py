import argparse
import pickle

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from dataset import create_batch_data
from model_tf import HierarchicalAttentionNetwork

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--dataset_file", default="20newsgroup.pkl", help="Destination file name for the dataset")
arg_parser.add_argument("--embedding_size", default=50, type=int, help="Embedding size")
arg_parser.add_argument("--gru_output_size", default=128, type=int, help="GRU output size")
arg_parser.add_argument("--batch_size", default=8, type=int, help="Number of examples in a batch")
arg_parser.add_argument("--num_epochs", default=30, type=int, help="Number of training epochs")
arg_parser.add_argument("--learning_rate", default=0.001, type=float, help="Optimizer learning rate")


def main(args):
    print("-- Loading dataset from: {}".format(args.dataset_file))
    with open(args.dataset_file, mode="rb") as in_file:
        dataset_data = pickle.load(in_file)

    num_classes = len(dataset_data["train"]["target_names"])
    num_tokens = len(dataset_data["vocab"])
    num_examples = len(dataset_data["train"]["data"])

    print("-- Dataset statistics")
    print("# examples: {}".format(num_examples))
    print("# classes: {}".format(num_classes))
    print("# tokens: {}".format(num_tokens))

    with tf.Session() as session:
        num_sentences_input = tf.placeholder(dtype=tf.int32, name="num_sentences_input")
        batch_size_input = tf.placeholder(dtype=tf.int32, name="batch_size_input")
        sentence_len_input = tf.placeholder(dtype=tf.int32, name="sentence_len_input")
        documents_input = tf.placeholder(dtype=tf.int32, shape=(None, None, None), name="documents_input")
        targets_input = tf.placeholder(dtype=tf.int32, shape=(None,), name="targets_input")

        params = {
            "num_tokens": num_tokens,
            "embedding_size": args.embedding_size,
            "gru_output_size": args.gru_output_size,
            "num_classes": num_classes
        }

        han_model = HierarchicalAttentionNetwork(documents_input,
                                                 num_sentences_input,
                                                 sentence_len_input,
                                                 batch_size_input,
                                                 params)

        loss_function = han_model.loss(targets_input)

        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        train_op = optimizer.minimize(loss_function)

        session.run(tf.global_variables_initializer())

        indexes = np.arange(0, num_examples)
        np.random.shuffle(indexes)

        num_batches = num_examples // args.batch_size
        epoch_loss = 0

        for e in range(args.num_epochs):
            batch_range = tqdm(range(0, num_examples, args.batch_size))
            for start_index in batch_range:
                batch_indexes = indexes[start_index: start_index + args.batch_size]
                batch_docs, max_num_sentences, max_sentence_len = create_batch_data(dataset_data["train"]["data"],
                                                                                    batch_indexes)
                batch_targets = dataset_data["train"]["target"][batch_indexes]

                loss, _ = session.run(
                    [loss_function, train_op],
                    {
                        batch_size_input: args.batch_size,
                        num_sentences_input: max_num_sentences,
                        documents_input: batch_docs,
                        targets_input: batch_targets,
                        sentence_len_input: max_sentence_len
                    }
                )
                epoch_loss += loss
                batch_range.set_postfix({"Loss": loss})

            epoch_loss /= num_batches
            batch_range.set_description("Last epoch loss: {}".format(epoch_loss))


if __name__ == "__main__":
    main(arg_parser.parse_args())
