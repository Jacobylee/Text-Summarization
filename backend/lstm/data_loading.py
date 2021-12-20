# -*- coding: utf-8 -*-
# Author: Xiangyu Li
# Data: 2021-11
import os
import torch
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

UNK = '<unk>'
PAD = '<pad>'
BOS = '<bos>'
EOS = '<eos>'

Article_TRUNCATE = 256
Summary_TRUNCATE = 200
# data
articles_dir = '/Users/jacob/Desktop/Term 3/217/summarization/backend/dataset/BBC News Summary/News Articles/'
summaries_dir = '/Users/jacob/Desktop/Term 3/217/summarization/backend/dataset/BBC News Summary/Summaries/'
classes = os.listdir(articles_dir)

contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                       "could've": "could have", "couldn't": "could not",

                       "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                       "hasn't": "has not", "haven't": "have not",

                       "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did",
                       "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                       "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
                       "I'm": "I am", "I've": "I have", "i'd": "i would",

                       "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am",
                       "i've": "i have", "isn't": "is not", "it'd": "it would",

                       "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is",
                       "let's": "let us", "ma'am": "madam",

                       "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                       "mightn't've": "might not have", "must've": "must have",

                       "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                       "needn't've": "need not have", "o'clock": "of the clock",

                       "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                       "sha'n't": "shall not", "shan't've": "shall not have",

                       "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                       "she'll've": "she will have", "she's": "she is",

                       "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                       "so've": "so have", "so's": "so as",

                       "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is",
                       "there'd": "there would",

                       "there'd've": "there would have", "there's": "there is", "here's": "here is",
                       "they'd": "they would", "they'd've": "they would have",

                       "they'll": "they will", "they'll've": "they will have", "they're": "they are",
                       "they've": "they have", "to've": "to have",

                       "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
                       "we'll've": "we will have", "we're": "we are",

                       "we've": "we have", "weren't": "were not", "what'll": "what will",
                       "what'll've": "what will have", "what're": "what are",

                       "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have",
                       "where'd": "where did", "where's": "where is",

                       "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is",
                       "who've": "who have",

                       "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
                       "won't've": "will not have",

                       "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                       "y'all": "you all",

                       "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",
                       "y'all've": "you all have",

                       "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                       "you'll've": "you will have",

                       "you're": "you are", "you've": "you have"}


# helper function
def tokenizer(text: str) -> list:
    """ Normalize a sentence.
        Args:
            text: a sentence
        :return:
            list of normalized tokens
    """
    text = ' '.join([contraction_mapping[i] if i in contraction_mapping.keys() else i for i in text.split()])
    text = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    stem = PorterStemmer()
    normalized = [stem.stem(wd) for wd in text if not (stem.stem(wd) in stop_words or len(stem.stem(wd)) <= 2)]
    return normalized


# vocab
def build_vocab(Articles_dir: str, Summaries_dir: str) -> tuple:
    """ build vocabulary.
        Args:
            Articles_dir: path of articles
            Summaries_dir: path of summaries
        :return:
            articles_dict: key:file_name value:list of tokens
            summaries_dict: key:file_name value:list of tokens
            article_vocab: article vocab set
            summary_vocab: summary vocab set
    """
    print("building vocabs....", end='')
    articles = {}
    summaries = {}
    article_vocab = set()
    summary_vocab = set()

    for cls in classes:
        if cls == '.DS_Store':
            continue
        files = os.listdir(Articles_dir + cls)
        for file in files:
            article_file_path = Articles_dir + cls + '/' + file
            summary_file_path = Summaries_dir + cls + '/' + file
            try:
                # articles
                with open(article_file_path, 'r') as f:
                    tokens = [line.rstrip() for line in f.readlines() if line.rstrip() != '']
                    # one-line article
                    article = ' '.join(tokens)
                    tokenized = tokenizer(article)
                    article_vocab.update(tokenized)
                    articles[cls+file] = tokenized
                with open(summary_file_path, 'r') as s:
                    tokens = ''.join([line.rstrip() for line in s.readlines()]).split('.')
                    # one-line summary
                    summary = ' '.join(tokens)
                    tokenized = tokenizer(summary)
                    summary_vocab.update(tokenized)
                    summaries[cls+file] = tokenized
            except:
                pass
    article_vocab = list(article_vocab)
    article_vocab += [PAD, UNK, EOS]
    summary_vocab = list(summary_vocab)
    summary_vocab += [PAD, UNK, EOS, BOS]
    print('done')
    return articles, summaries, article_vocab, summary_vocab


def build_index(dataset: set) -> dict:
    """ build vocabulary index.
        Args:
            dataset: set of dataset
        :return:
            stoi: tokens index
    """
    print("building index...", end='')
    stoi = {word: i for i, word in enumerate(dataset)}
    print('done')
    return stoi


# vectorized data
def convert_tensor(path: str, index: dict, truncate: int, is_output: bool):
    """ convert seq to vector.
        Args:
            path: text
            index: vocab
            truncate：int
        :return:
            vector: data tensor
    """
    with open(path, 'r') as f:
        result = [index[PAD]] * truncate
        out_seq = []

        tokens = [line.rstrip() for line in f.readlines() if line.rstrip() != '']
        text = ' '.join(tokens)
        seq = tokenizer(text)
        if is_output:
            seq = [BOS] + seq + [EOS]
        else:
            seq = seq + [EOS]
        for tok in seq:
            if tok in index:
                out_seq.append(index[tok])
            else:
                out_seq.append(index[UNK])
        if len(out_seq) >= truncate:
            new_vector = out_seq[:truncate]
            new_vector[-1] = index[EOS]
            return new_vector, out_seq
        else:
            result[:len(out_seq)] = out_seq
            return result, out_seq


def build_data_vectors(Articles_dir: str, Summaries_dir: str, article_index: dict, summary_index: dict) -> tuple:
    """ convert seq to vector.
        Args:
            Articles_dir: Articles path
            Summaries_dir: Summaries path
            article_index: article index dict
            summary_index: summary index dict
        :return:
            vector: dataset vector [(article vector, summary vector), (), ...]
    """
    print("building data vectors...", end='')
    src_vectors = {}
    tgt_vectors = {}

    src_vectors_untruncated = {}
    tgt_vectors_untruncated = {}
    for cls in classes:
        if cls == '.DS_Store':
            continue
        files = os.listdir(Articles_dir + cls)
        for file in files:
            article_file_path = Articles_dir + cls + '/' + file
            summary_file_path = Summaries_dir + cls + '/' + file
            try:
                # articles
                src, src_original = convert_tensor(article_file_path, article_index, Article_TRUNCATE, is_output=False)
                src_vectors[cls+file] = src
                src_vectors_untruncated[cls+file] = src_original
                # summaries
                tgt, tgt_original = convert_tensor(summary_file_path, summary_index, Summary_TRUNCATE, is_output=True)
                tgt_vectors[cls+file] = tgt
                tgt_vectors_untruncated[cls+file] = tgt_original
            except:
                pass
    print('done')
    return src_vectors, tgt_vectors, src_vectors_untruncated, tgt_vectors_untruncated


a_filename_tokens, s_filename_tokens, src_itos, tgt_itos = build_vocab(articles_dir, summaries_dir)
print(tgt_itos)
