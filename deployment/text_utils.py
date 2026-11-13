"""Text cleaning shared by the notebooks and the web app.

The saved model pipeline refers to preprocess() by name, so this file
must sit next to app.py and model.joblib.
"""
import re

import simplemma
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Words that appear in nearly every abstract and say nothing about the topic.
EXTRA_STOPWORDS = {
    "abstract", "summary", "paper", "article", "study", "research", "result",
    "results", "use", "using", "used", "show", "also", "may", "one", "two",
    "elsevier", "copyright", "rights", "reserved", "author", "authors",
}
STOPWORDS = set(ENGLISH_STOP_WORDS) | EXTRA_STOPWORDS

TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"(https?://|www\.)\S+")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
DOI_RE = re.compile(r"\b10\.\d{4,9}/\S+")
BRACKET_RE = re.compile(r"\[[^\]]{0,40}\]")
NON_LETTER_RE = re.compile(r"[^a-z\s]")
SPACE_RE = re.compile(r"\s+")


def clean_text(text):
    """Strip markup, links, emails, DOIs, bracketed refs, digits and punctuation."""
    text = str(text)
    text = TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = DOI_RE.sub(" ", text)
    text = BRACKET_RE.sub(" ", text)
    text = text.lower()
    text = text.replace("-", " ")
    text = NON_LETTER_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def preprocess(text):
    """Clean, tokenise, drop stop words and short tokens, then lemmatise."""
    tokens = []
    for tok in clean_text(text).split():
        if tok in STOPWORDS:
            continue
        lemma = simplemma.lemmatize(tok, lang="en")
        if lemma in STOPWORDS:
            continue
        if len(lemma) < 3 and lemma != "ai":
            continue
        tokens.append(lemma)
    return " ".join(tokens)
