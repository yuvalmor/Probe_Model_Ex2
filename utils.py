from functools import reduce
from math import log2
from sys import float_info

voc_size = 300000


def perplexity(model, data):
    probs = [model.get_prob(vald_word) for vald_word in data]
    probs_log = [
        log2(word_prob) if word_prob > 0 else log2(float_info.epsilon)
        for word_prob in probs
    ]
    sum_probs = reduce(lambda a, b: a + b, probs_log)
    power_val = -sum_probs / len(data)
    return 2 ** power_val
