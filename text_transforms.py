from nltk.stem.snowball import RussianStemmer
from collections import Counter

import json
import re
import numpy as np


def __cleaning(word: str) -> bool:
    if not word:
        return False
    if re.match(r'.*\d+.*', word):
        return False
    return True


def _get_words(text: str) -> list[str]:
    uncleaned = re.findall(r'[Нн]а самом деле|[Вв] общем|\w+-\w+|\w+', text)
    cleaned = filter(__cleaning, uncleaned)
    normalized = map(lambda word: word.lower(), cleaned)
    stemmer = RussianStemmer()
    stemmed = map(lambda word: stemmer.stem(word), normalized)
    return list(stemmed)


def get_features(request: dict) -> dict:
    """

    :param request: dict - dict-container for text from user. It essential to provide
        text via 'message' key
    :return: dict - features vector provided via 'features'
    """
    if 'message' not in request:
        raise ValueError('No "message" field in the JSON')
    text = request.get('message')
    words = _get_words(text)
    document_words = Counter(words)

    with open('corpus.json', 'r') as f:
        corpus_info = json.load(f)
        total_documents = corpus_info['documents_number']
        key_words = corpus_info['key_words']
        corpus_words = Counter(corpus_info['corpus'])

    corpus_words.update(set(words).intersection(set(key_words)))

    features = np.zeros(shape=(len(key_words), ), dtype=float)
    for ind, key in enumerate(key_words):
        tf = document_words[key] / document_words.total()
        idf = np.log(total_documents / corpus_words[key])
        features[ind] = tf * idf

    return {
        'features': features.tolist()
    }
