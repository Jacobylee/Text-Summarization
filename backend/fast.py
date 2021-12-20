# Use uvicorn to run this:
# uvicorn fast:app --reload
# See documentation and a test interface for the service once it's running:
# http://localhost:8000/docs
from fastapi import FastAPI
from pydantic import BaseModel
from weighted import Summarizer
from pagerank import PageRank
# package baseline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

app = FastAPI()

_summarizer1 = Summarizer()  # change mode
_summarizer2 = PageRank()


class SummarizerRequest(BaseModel):
    text: str


class SummarizerResponse(BaseModel):
    text: str
    summary: str


@app.get("/normal_summary")
async def summart_get(text: str) -> SummarizerResponse:
    _summarizer1.train(text)
    summary = _summarizer1.forward(text)
    return SummarizerResponse(text=text, summary=summary)


@app.get("/pagerank_summary")
async def summary_get(text: str, line: int) -> SummarizerResponse:
    if line >= 5:
        line = 5
    _summarizer2.train(text)
    summary = _summarizer2.extract_summary(line)
    return SummarizerResponse(text=text, summary=summary)


@app.get("/lexrank_summary")
async def summart_get(text: str, line: int) -> SummarizerResponse:
    if line >= 5:
        line = 5
    # For Strings
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    # Using LexRank
    summarizer = LexRankSummarizer()
    # Summarize the document with k sentences
    summaries = summarizer(parser.document, line)
    summary = ''
    for sentence in summaries:
        summary += str(sentence) + '. '
    return SummarizerResponse(text=text, summary=summary)
