# extra_features.py
import numpy as np
import string
from sklearn.base import BaseEstimator, TransformerMixin

emoji_set = set(
    "😀😁😂🤣😃😄😅😆😉😊😋😎😍😘😗😙😚🙂🤗🤩🤔🤨😐😑😶🙄😏😣😥"
    "😮🤐😯😪😫😴😌🤓😛😜😝🤤😒😓😔😕🙃🤑😲☹️🙁😖😞😟😤😢😭"
    "😦😧😨😩🤯😬😰😱😳🤪😵😡😠🤬"
)

class ExtraFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        # ensure texts is an iterable of strings
        processed = [t if isinstance(t, str) else "" for t in texts]

        emoji_count = np.array([sum(1 for ch in t if ch in emoji_set) for t in processed]).reshape(-1, 1)
        punctuation_ratio = np.array([sum(1 for c in t if c in string.punctuation) / (len(t) + 1) for t in processed]).reshape(-1, 1)
        digit_ratio = np.array([sum(1 for c in t if c.isdigit()) / (len(t) + 1) for t in processed]).reshape(-1, 1)
        avg_word_len = np.array([
            np.mean([len(w) for w in t.split()]) if len(t.split()) > 0 else 0
            for t in processed
        ]).reshape(-1, 1)

        return np.hstack([emoji_count, punctuation_ratio, digit_ratio, avg_word_len])
