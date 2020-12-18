import time
from pathlib import Path
import gensim

def get_model(path):
    model = gensim.models.KeyedVectors.load_word2vec_format(
        path, binary=True, limit=500000
    )
    return model

def make_clues(model, positives, negatives):
    return model.most_similar(positive = positives, negative = negatives, restrict_vocab=50000,topn=20)
