import pandas as pd
import numpy as np
import torch
import os

def make_masked_copy(filepath):
    """ Makes a copy of a data set in which all words that are distinct for a class are masked. """

    df = pd.read_csv(filepath, sep="\t", header=None, names=["id", "label", "alpha", "text"])
    df.text = df.text.astype(str)

    text_1 = " ".join(df[df.label == 1].text.values)
    text_2 = " ".join(df[df.label == 0].text.values)

    vocab_1 = set(text_1.split())
    vocab_2 = set(text_2.split())

    unique_words = (vocab_1 - vocab_2).union(vocab_2 - vocab_1)
    df.text = df.text.apply(lambda x: mask_sent(x, unique_words))

    new_fp = filepath[:-4] + "_masked.tsv"
    df.to_csv(new_fp, sep="\t", index=False, header=False)
	
	
def make_classification_dataset(dataset_dir, experiment_dir):
    """ Creates a balanced time classification dataset from a diachronic LSCD dataset. """

    prep_dir = experiment_dir + "preprocessed_texts/"
    os.makedirs(prep_dir, exist_ok=True)

    with open(dataset_dir + "corpus19.txt", "r") as fh: # with open(dataset_dir + "c1.txt", "r") as fh:
        sents_c1 = fh.read().splitlines()

    with open(dataset_dir + "corpus20.txt", "r") as fh: # with open(dataset_dir + "c2.txt", "r") as fh:
        sents_c2 = fh.read().splitlines()

    # detemine thresholds
    n_samples_per_class = min(len(sents_c1), len(sents_c2)) #(min(len(sents_c1), len(sents_c2)) // 10_000) * 10_000
    n_train = int(n_samples_per_class * 0.8 * 2)
    n_test = n_samples_per_class * 2 - n_train


    # collect samples for each class
    df_0 = pd.DataFrame({"text": sents_c1, "label": 0}).sample(n=n_samples_per_class, replace=False)
    df_1 = pd.DataFrame({"text": sents_c2, "label": 1}).sample(n=n_samples_per_class, replace=False)

    # sample train and test data for each label without overlap
    perm = np.random.permutation(n_samples_per_class)
    train_df = pd.concat([df_0.iloc[perm[:(n_train // 2)]], df_1.iloc[perm[:(n_train // 2)]]], ignore_index=True)
    test_df = pd.concat([df_0.iloc[perm[(n_train // 2):]], df_1.iloc[perm[(n_train // 2):]]], ignore_index=True)

    # check balance of labels
    assert np.all(train_df.label.value_counts() / n_train == test_df.label.value_counts() / n_test), "Classification dataset is not balanced!"

    # shuffle train and test data
    train_df = train_df.sample(frac=1)
    test_df = test_df.sample(frac=1)

    for df in [train_df, test_df]:

        # remove year that is at beginning of sentence in some datasets
        df["text"] = df["text"].str.rsplit("\t", expand=True)[0]

        # create dummy columns to conform with BERT dataset format
        df["alpha"] = ["a"] * len(df.index)
        df["id"] = range(len(df.index))

    train_df[["id", "label", "alpha", "text"]].to_csv(prep_dir + "train.tsv", sep="\t", index=False, header=False)
    test_df[["id", "label", "alpha", "text"]].to_csv(prep_dir + "test.tsv", sep="\t", index=False, header=False)

    make_masked_copy(prep_dir + "train.tsv")
    make_masked_copy(prep_dir + "test.tsv")