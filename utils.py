""" Yuval mor, Roey Fuchs 205380173, 205415342 """
from functools import reduce
from math import log2
from sys import float_info

voc_size = 300000  # as given


def perplexity(model, data):
    """ calculate input model perplexity with input data """
    probs = [model.get_prob(word) for word in data]  # get word's probability
    probs_log = [
        log2(word_prob) if word_prob > 0 else log2(float_info.epsilon)
        for word_prob in probs
    ]  # log the probabilities. using epsilon when the probability is 0
    sum_probs = reduce(lambda a, b: a + b, probs_log)  # sum all
    power_val = (-1 * sum_probs) / len(probs_log)  # divide by n and neg all
    return 2 ** power_val


def sum_model_probs(model, uniq_words):
    """this function will return test the model. it will return sum of
    probabilities of all input words list. test will pass if the return value
    is ~1"""
    sum_probs = 0
    for word in uniq_words:
        sum_probs += model.get_prob(word)
    sum_probs += (voc_size - len(uniq_words)) * model.get_prob_by_word_freq(0)
    return sum_probs
