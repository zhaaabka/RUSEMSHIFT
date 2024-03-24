import pandas as pd
import numpy as np
import torch
import os


def mask_sent(sent, uniques):
    """ Replaces unique words with [MASK]-token. """

    return " ".join(["[MASK]" if word in uniques else word for word in sent.split()])


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
	
	
def make_classification_dataset(c1_path, c2_path, c3_path, experiment_dir):

    prep_dir = experiment_dir + "preprocessed_texts/"
    os.makedirs(prep_dir, exist_ok=True)

    with open(c1_path, "r") as fh:
        sents_c1 = fh.read().splitlines()

    with open(c2_path, "r") as fh:
        sents_c2 = fh.read().splitlines()
        
    with open(c3_path, "r") as fh:
        sents_c3 = fh.read().splitlines()

    n_samples_per_class = min([len(sents_c1), len(sents_c2), len(sents_c3)])
    n_train = int(n_samples_per_class * 0.8 * 3)
    n_test = n_samples_per_class * 3 - n_train

    df_0 = pd.DataFrame({"text": sents_c1, "label": 0}).sample(n=n_samples_per_class, replace=False)
    df_1 = pd.DataFrame({"text": sents_c2, "label": 1}).sample(n=n_samples_per_class, replace=False)
    df_2 = pd.DataFrame({"text": sents_c3, "label": 2}).sample(n=n_samples_per_class, replace=False)

    perm = np.random.permutation(n_samples_per_class)
    train_df = pd.concat([df_0.iloc[perm[:(n_train // 2)]], df_1.iloc[perm[:(n_train // 2)]], df_2.iloc[perm[:(n_train // 2)]]], ignore_index=True)
    test_df = pd.concat([df_0.iloc[perm[(n_train // 2):]], df_1.iloc[perm[(n_train // 2):]], df_2.iloc[perm[:(n_train // 2)]]], ignore_index=True)

    train_df = train_df.sample(frac=1)
    test_df = test_df.sample(frac=1)

    for df in [train_df, test_df]:
        df["alpha"] = ["a"] * len(df.index)
        df["id"] = range(len(df.index))

    train_df[["id", "label", "alpha", "text"]].to_csv(prep_dir + "train.tsv", sep="\t", index=False, header=False)
    test_df[["id", "label", "alpha", "text"]].to_csv(prep_dir + "test.tsv", sep="\t", index=False, header=False)

    make_masked_copy(prep_dir + "train.tsv")
    make_masked_copy(prep_dir + "test.tsv")
