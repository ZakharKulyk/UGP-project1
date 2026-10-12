import numpy as np

def majority_baseline_predict(y_train):
    """
    Baseline: always predict the majority training class.
    """
    counts = np.bincount(y_train)
    majority_class = np.argmax(counts)
    return np.full(shape=len(y_train), fill_value=majority_class, dtype=np.int64)