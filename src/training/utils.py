import numpy as np
from scipy.stats import entropy



def batch_entropy(batched_probs, base=None):
    return np.apply_along_axis(lambda x: entropy(x, base=base), axis=1, arr=batched_probs)