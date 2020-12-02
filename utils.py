from functools import reduce
from math import log2
from sys import float_info

voc_size = 300000


def perplexity(model, data):
    probs = [model.get_prob(word) for word in data]
    probs_log = [
        log2(word_prob) if word_prob > 0 else log2(float_info.epsilon)
        for word_prob in probs
    ]
    sum_probs = reduce(lambda a, b: a + b, probs_log)
    power_val = (-1 * sum_probs) / len(probs_log)
    return 2 ** power_val


def sum_model_probs(model, uniq_words):
    sum = 0
    for word in uniq_words:
        sum += model.get_prob(word)
    sum += (voc_size - len(uniq_words)) * model.get_prob_by_word_freq(0)
    return sum
