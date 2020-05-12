# coding: utf-8

import re, random, io, nltk
import gluonnlp as nlp
import numpy as np
import mxnet as mx
from nltk import word_tokenize
nltk.download('punkt')

def load_tsv_to_array(fname):
    """
    Inputs: file path
    Outputs: list/array of 3-tuples, each representing a data instance
    """
    arr = []
    with io.open(fname, 'r') as fp:
        for line in fp:
            els = line.split('\t')
            els[3] = els[3].strip().lower()
            els[2] = int(els[2])
            els[1] = int(els[1])
            arr.append(tuple(els))
    return arr

relation_types = [
    "Component-Whole",
    "Component-Whole-Inv",
    "Instrument-Agency",
    "Instrument-Agency-Inv",
    "Member-Collection",
    "Member-Collection-Inv",
    "Cause-Effect",
    "Cause-Effect-Inv",
    "Entity-Destination",
    "Entity-Destination-Inv",
    "Content-Container",
    "Content-Container-Inv",
    "Message-Topic",
    "Message-Topic-Inv",
    "Product-Producer",
    "Product-Producer-Inv",
    "Entity-Origin",
    "Entity-Origin-Inv",
    "Other"
    ]


###    - Parse the input data by getting the word sequence and the argument POSITION IDs for e1 and e2
###    [[w_1, w_2, w_3, .....], [pos_1, pos_2], [label_id]]  for EACH data instance/sentence/argpair
def load_dataset(file, test_file, max_length=100):
    """
    Inputs: training file in TSV format. Split the file later. Cross validation
    Outputs: vocabulary (with attached embedding), training, validation and test datasets ready for neural net training
    """
    train_array = load_tsv_to_array(file)
    # val_array   = load_tsv_to_array(val_file)
    test_array  = load_tsv_to_array(test_file)
    
    # vocabulary  = build_vocabulary(train_array, val_array, test_array)
    # vocabulary.reserved_tokens.extend(['e1_start', 'e1_end', 'e2_start', 'e2_end'])
    vocabulary = build_vocabulary(train_array, test_array)
    dataset = preprocess_dataset(train_array, vocabulary, max_length)
    # train_dataset = preprocess_dataset(train_array, vocabulary, max_length)
    # val_dataset = preprocess_dataset(val_array, vocabulary, max_length)
    test_dataset = preprocess_dataset(test_array, vocabulary, max_length)

    data_transform = BasicTransform(relation_types, max_length)
    return vocabulary, dataset, test_dataset, data_transform

def tokenize(txt):
    """
    Tokenize an input string. Something more sophisticated may help . . . 
    """
    return word_tokenize(txt)


def build_vocabulary(tr_array, tst_array):
    """
    Inputs: arrays representing the training, validation and test data
    Outputs: vocabulary (Tokenized text as in-place modification of input arrays or returned as new arrays)
    """
    all_tokens = []
    tr_array, tokens = _get_tokens(tr_array)
    all_tokens.extend(tokens)
    # val_array, tokens = _get_tokens(val_array)
    # all_tokens.extend(tokens)
    tst_array, tokens = _get_tokens(tst_array)
    all_tokens.extend(tokens)
    counter = nlp.data.count_tokens(all_tokens)
    vocab = nlp.Vocab(counter)
    return vocab


def _get_tokens(array):
    all_tokens = []
    for i, instance in enumerate(array):
        label, e1, e2, text = instance
        tokens = text.split(" ")
        tokens.insert(e2 + 1, "e2_end")
        tokens.insert(e2, "e2_start")
        tokens.insert(e1 + 1, "e1_end")
        tokens.insert(e1, "e1_start")
        text = ' '.join(tokens)
        tokens = tokenize(text)
        inds = [tokens.index("e1_start")+1, tokens.index("e2_start")+1]
        tokens = [token for token in tokens if token not in["e1_start", "e2_start", "e1_end", "e2_end"]]
        inds[0] = inds[0] - 1
        inds[1] = inds[1] - 3
        array[i] = (label, inds[0], inds[1], tokens)  ## IN-PLACE modification of tr_array
        all_tokens.extend(tokens)
    return array, all_tokens


def _preprocess(x, vocab, max_len):
    """
    Inputs: data instance x (tokenized), vocabulary, maximum length of input (in tokens)
    Outputs: data mapped to token IDs, with corresponding label
    """
    label, ind1, ind2, text_tokens = x
    data = vocab[text_tokens]   ## map tokens (strings) to unique IDs
    data = data[:max_len]       ## truncate to max_len

    return label, ind1, ind2, data

def preprocess_dataset(dataset, vocab, max_len):
    preprocessed_dataset = [ _preprocess(x, vocab, max_len) for x in dataset]
    return preprocessed_dataset


class BasicTransform(object):
    """
    This is a callable object used by the transform method for a dataset. It will be
    called during data loading/iteration.  

    Parameters
    ----------
    labels : list string
        List of the valid strings for classification labels
    max_len : int, default 32
        Maximum sequence length - longer seqs will be truncated and shorter ones padded
    
    """
    def __init__(self, labels, max_len=32):
        self._max_seq_length = max_len
        self._label_map = {}
        for (i, label) in enumerate(labels):
            self._label_map[label] = i
        self._label_map['?'] = i+1
    
    def __call__(self, label, ind1, ind2, data):
        label_id = self._label_map[label]
        padded_data = data + [0] * (self._max_seq_length - len(data))
        inds = mx.nd.array([ind1, ind2])
        return mx.nd.array(padded_data, dtype='int32'), inds, mx.nd.array([label_id], dtype='int32')


def split_file(file):
    with open(file) as f:
        lines = f.readlines()
    random.shuffle(lines)
    train_size = int(len(lines) * 0.8)
    val_size = int(len(lines) * 0.2)
    with open("../train.tsv", "w") as f:
        for line in lines[0:train_size]:
            f.write(line)
    with open("../val.tsv", "w") as f:
        for line in lines[-val_size:]:
            f.write(line)


if __name__=="__main__":
    # load_tsv_to_array("../semevalTrain.tsv")
    split_file("../semevalTrain.tsv")
    train_file = "../train.tsv"
    test_file = "../test.tsv"
    val_file = "../val.tsv"
    # load_dataset(train_file, val_file, test_file, 100)
