from pathlib import Path
import pytest
from src.cluefinder import cluefinder


@pytest.fixture
def gensim_word2vec():
    path = Path(__file__).resolve()
    model = cluefinder.get_model(path.parents[2] / 'data' / 'GoogleNews-vectors-negative300.bin') 
    return model

def test_get_model():
    path = Path(__file__).resolve()
    model = cluefinder.get_model(path.parents[2] / 'data' / 'GoogleNews-vectors-negative300.bin')


def test_make_clues(gensim_word2vec):
    model = gensim_word2vec
    positives = ['Earth','Moon']
    negatives = ['planet']
    clues = model.most_similar(positive = positives, negative = negatives, restrict_vocab=50000,topn=20)
    assert clues[0][0] == 'Lunar'