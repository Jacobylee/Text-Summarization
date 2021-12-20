import os

from backend.pagerank import PageRank
from backend.weighted import Summarizer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
# package baseline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

articles_dir = '/Users/jacob/Desktop/Term 3/217/summarization/backend/dataset/BBC News Summary1/News Articles/'
summaries_dir = '/Users/jacob/Desktop/Term 3/217/summarization/backend/dataset/BBC News Summary1/Summaries/'
classes = os.listdir(articles_dir)


def evaluation(model):
    result = 0
    dataset_size = 0
    cc = SmoothingFunction().method4
    for cls in classes:
        if cls == '.DS_Store':
            continue
        files = os.listdir(articles_dir + cls)
        for file in files:
            dataset_size += 1
            article_file_path = articles_dir + cls + '/' + file
            summary_file_path = summaries_dir + cls + '/' + file
            with open(article_file_path, 'r') as f:
                lines = [str(line.rstrip()) for line in f.readlines() if line.rstrip() != '']
                news = '. '.join(lines)
                if model == 'Weighted':
                    _summarizer = Summarizer()
                    _summarizer.train(news)
                    summary = _summarizer.forward(news)
                    src_tokens = summary.split()
                elif model == 'LexRanking':
                    # For Strings
                    parser = PlaintextParser.from_string(news, Tokenizer("english"))
                    # Using LexRank
                    summarizer = LexRankSummarizer()
                    # Summarize the document with 2 sentences
                    summaries = summarizer(parser.document, 6)
                    summary = ''
                    for sentence in summaries:
                        summary += str(sentence) + '. '
                    src_tokens = summary.split()
                elif model == 'PageRank':
                    # 3: 0.32930304577692276
                    # 4: 0.39358131272363944
                    # 5: 0.4300763629583866
                    # 6: 0.4481261422079907
                    _pagerank = PageRank()
                    _pagerank.train(news)
                    summary = _pagerank.extract_summary(5)
                    src_tokens = summary.split()
            with open(summary_file_path, 'r') as f:
                lines = [line.rstrip().split() for line in f.readlines() if line.rstrip() != '']
                tgt_tokens = []
                for i in lines:
                    tgt_tokens += i
            bleu_score = sentence_bleu([src_tokens], tgt_tokens, smoothing_function=cc)
            result += bleu_score
    print(dataset_size)
    print(model + ' model average BLEU score: ', result/dataset_size)


evaluation('Weighted')
evaluation('LexRanking')
evaluation('PageRank')
