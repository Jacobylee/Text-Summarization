# -*- coding: utf-8 -*-
# Author: Xiangyu Li
# Date: 2021-11
import click
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu



class Summarizer:
    def __init__(self):
        self.stem = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))
        self.weighted_table = {}
        self.average_score = 0.0

    @staticmethod
    def text_reader(news: str):
        """ Split a article to sentences.
            Args:
                news: an original article
        """
        return news.split('. ')

    def word_tokenize(self, text: str) -> list:
        """ Normalize a sentence.
            Args:
                text: a sentence
        """
        unnormalized = text.split()
        normalized = [self.stem.stem(wd) for wd in unnormalized if self.stem.stem(wd) not in self.stop_words]
        return normalized

    def calculate_average_score(self, sentence_weight) -> None:
        """ Calculating the average score for the sentences.
            Args:
                sentence_weight: list of weighted sentence
        """
        # Calculating the average score for the sentences
        sum_values = 0
        for entry in sentence_weight:
            sum_values += sentence_weight[entry]
        # Getting sentence average value from source text
        average_score = (sum_values / len(sentence_weight))
        self.average_score = average_score

    def train(self, text_string: str) -> None:
        """ Trains model to make a weighted table.
            Args:
                text_string: an original article
        """
        word_counter = Counter()
        # list of sentence
        sentences = self.text_reader(text_string)
        for sentence in sentences:
            words = self.word_tokenize(sentence)
            word_counter.update(words)
        highest = word_counter.most_common(1)[0][1]
        # weighted
        self.weighted_table = {word: word_counter[word]/highest for word in word_counter}

    def forward(self, news: str) -> str:
        """ Calculating the summary.
            Args:
                news: an original article
        """
        scores = {}
        sentences = self.text_reader(news)
        for i in range(len(sentences)):
            score = 0
            normalized = self.word_tokenize(sentences[i])
            if len(normalized) == 0:
                continue
            for token in normalized:
                score += self.weighted_table[token]
            scores[i] = score/len(normalized)
        # calculate average score
        self.calculate_average_score(scores)
        abstract = ''
        for sentence in range(len(sentences)):
            if sentence in scores:
                if scores[sentence] >= self.average_score:
                    abstract += ' ' + sentences[sentence] + '.'
        return abstract[1:-1]


@click.command()
@click.argument('input_article', type=str)
@click.option('--label', help='abstract wrote by human')
def main(input_article: str, label: str) -> None:
    # python weighted.py ARTICLE --label HUMAN_LABEL
    summarizer = Summarizer()
    summarizer.train(input_article)
    summary = summarizer.forward(input_article)
    print("-"*150)
    print("Summary:", summary)
    if label:
        print("-" * 150)
        print("Human Abstract:", label)
        print("-" * 150)
        reference = summarizer.word_tokenize(label)
        test = summarizer.word_tokenize(summary)
        score = sentence_bleu([reference], test)
        print(score)


if __name__ == '__main__':
    main()
