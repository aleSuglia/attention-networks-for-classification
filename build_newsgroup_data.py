import argparse
import pickle

import spacy

from dataset import process_20newsgroup_dataset


def main(args):
    nlp = spacy.load("en")
    train_dataset, test_dataset = process_20newsgroup_dataset(nlp)

    print("-- Saving processed dataset to: {}".format(args.dataset_file))
    with open(args.dataset_file, mode="wb") as out_file:
        pickle.dump({
            "vocab": nlp.vocab.strings,
            "train": train_dataset,
            "test": test_dataset
        }, out_file)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dataset_file", help="Destination file name for the processed dataset")
    main(arg_parser.parse_args())
