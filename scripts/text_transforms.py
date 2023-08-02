from nltk.stem.snowball import RussianStemmer
from collections import Counter
from pathlib import Path

import os
import sys
import inspect
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


def get_features(data: str | list[str],
                 mode: str,
                 fp: str) -> np.ndarray:
    """
    :param fp: defines where corpus is located
    :param mode: parameter for defining behaviour
    :param data: str | list[str]. Text for encoding with tf-idf
    :return: np.ndarray. Array with required features
    """

    if isinstance(data, str):
        data = [data]
    if isinstance(data, list):
        with open(fp, 'r') as f:
            corpus_info = json.load(f)
            total_documents = corpus_info['documents_number']
            key_words = corpus_info['key_words']
            corpus_words = Counter(corpus_info['corpus'])

        features = np.zeros(shape=(len(data), len(key_words)), dtype='float32')
        for text_ind, text in enumerate(data):
            words = _get_words(text)
            document_words = Counter(words)
            if document_words.total():
                common_words = set(key_words).intersection(document_words.keys()) if mode == 'test' else {}
                corpus_words.update(common_words)

                for feature_ind, key in enumerate(key_words):
                    tf = document_words[key] / document_words.total()
                    idf = np.log(total_documents / corpus_words[key])
                    features[text_ind][feature_ind] = tf * idf
                corpus_words.subtract(common_words)
        return features
    raise ValueError(f'Incorrect type of data provided: {type(data)}')


if __name__ == '__main__':
    """
        It's essential to provide text through 'message' key
        in JSON file!
        The output file contains 'features' key with list of
        feature-numbers.
        
        Before running the script you need to provide 2 CL arguments:
        source file and destination file names with the corresponding order.
    """
    cur_dir = Path(inspect.stack()[0][1]).parent
    src_file = os.path.join(cur_dir, '../JSONs/' + sys.argv[1])
    dst_file = os.path.join(cur_dir, '../JSONs/' + sys.argv[2])
    corpus_file = os.path.join(cur_dir, '../JSONs/corpus.json')

    with open(src_file, 'r') as f:
        text = json.load(f)['message']
    with open(dst_file, 'w') as f:
        to_write = {
            'features': get_features(text, mode='test', fp=corpus_file).tolist()[0]
        }
        json.dump(to_write, f)
