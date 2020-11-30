import sys  # for epsilon
from functools import reduce
from math import log2

from utils import perplexity, voc_size


class lidstone:
    def __init__(self, counter_words, seq, lambda_value):
        self.counter_words = counter_words
        self.seq = seq
        self.lambda_value = lambda_value

    def get_prob(self, word):
        return (self.counter_words[word] + self.lambda_value) / (
            len(self.seq) + self.lambda_value * voc_size
        )

    def get_perplexity(self, data):
        return perplexity(self, data)
