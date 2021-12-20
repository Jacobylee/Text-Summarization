# -*- coding: utf-8 -*-
# Author: Xiangyu Li
# Date: 2021-11
import numpy as np
import pandas as pd
from pandas.core import series
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# nltk.download('punkt')
# nltk.download('stopwords')


def obtain_word_embeddings(path: str) -> dict:
    """
        Build Embedding Table
    """
    word_embeddings = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            values = line.split()  # Split data
            word = values[0]  # Get its name
            coefs = np.asarray(values[1:], dtype='float32')  # Get its coefficient
            word_embeddings[word] = coefs
    return word_embeddings


embeddings_table = obtain_word_embeddings('/Users/jacob/Desktop/Term 3/217/summarization/backend/dataset/glove.6B'
                                          '.100d.txt')


class PageRank:
    def __init__(self):
        self.stop_words = stopwords.words('english')

        self.raw_sentences = []
        self.clean_sentences = None
        self.embedding_sentences = None

        self.ranked_sentences = None

    def train(self, news: str):
        self.normalization(news)  # normalized data
        self.get_sentence_vector()  # convert data to embedding vector
        self.forward()

    @staticmethod
    def sentence_tokenize(sent: str) -> list:
        """
            split sentences
            :param sent: line
            :return: lst: list of sentence
        """
        lst = []
        tmp_lst = []
        for i in range(len(sent)):
            if not (len(tmp_lst) == 0 and sent[i] == ' '):
                tmp_lst.append(sent[i])
            if i == len(sent) - 1 or (sent[i] == '.' and (not (sent[i + 1].isdigit()) or sent[i + 1] == " ")):
                lst.append(''.join(tmp_lst))
                tmp_lst = []
        return lst

    def normalization(self, articles: str) -> series:
        """
            flush sentences by lower, remove stop words and spaces
            :param articles: sentences
            :return: sentences: list of lower sentences without space
        """
        # remove space
        sentences = []
        articles_lists = articles.split('\n')
        for s in articles_lists:
            sentences.extend(self.sentence_tokenize(s))
        self.raw_sentences = sentences
        # lower
        sentences = [s.lower() for s in sentences]
        # remove stop word
        clean_stopword_sentences = []
        for sen in sentences:
            clean_stopword_sentences.append(" ".join([i for i in sen.split() if i not in self.stop_words]))
        # remove unchar
        clean_sentences = pd.Series(clean_stopword_sentences).str.replace("[^a-zA-Z]", " ", regex=True)
        self.clean_sentences = clean_sentences

    def get_sentence_vector(self):
        """
            Build Embedding Vectors
        """
        sentence_vectors = []
        for i in self.clean_sentences:
            if len(i) != 0 and len(i.split()) != 0:
                v = sum([embeddings_table.get(w, np.zeros((100,))) for w in i.split()]) / (
                    len(i.split()))
            else:
                v = np.zeros((100,))
            sentence_vectors.append(v)
        self.embedding_sentences = np.array(sentence_vectors)

    def forward(self):
        """
            Build ranked table by relative score
        """
        # similarity matrix
        sim_mat = np.zeros([len(self.raw_sentences), len(self.raw_sentences)])
        for i in range(len(self.raw_sentences)):
            for j in range(len(self.raw_sentences)):
                if i != j:
                    sim_mat[i][j] = \
                        cosine_similarity(self.embedding_sentences[i].reshape(1, 100),
                                          self.embedding_sentences[j].reshape(1, 100))[0, 0]
        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)
        ranked_sentences = sorted(((scores[i], i) for i, s in enumerate(self.raw_sentences)), reverse=True)
        self.ranked_sentences = ranked_sentences

    def extract_summary(self, k):
        """
        :param k: Top k relative sentences
        :return: summary
        """
        # Specify number of sentences to form the summary
        result_lst = []
        # Generate summary
        for i in range(k):
            result_lst.append(self.ranked_sentences[i][1])
        return '. '.join([self.raw_sentences[i] for i in sorted(result_lst)])
