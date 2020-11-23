import sys  # for epsilon
from functools import reduce
from math import log2

voc_size = 300000


class lidstone:
    def __init__(self, counter_words, seq, lambda_value):
        self.counter_words = counter_words
        self.seq = seq
        self.lambda_value = lambda_value

    def get_prob(self, word):
        return (self.counter_words[word] + self.lambda_value) / (
            len(self.seq) + self.lambda_value * voc_size
        )

    def get_perplexity(self, vald):
        probs = [self.get_prob(vald_word) for vald_word in vald]
        probs_log = [
            log2(word_prob) if word_prob > 0 else log2(sys.float_info.epsilon)
            for word_prob in probs
        ]
        sum_probs = reduce(lambda a, b: a + b, probs_log)
        power_val = -sum_probs / len(vald)
        print(self.lambda_value, power_val, 2 ** power_val)
        return 2 ** power_val
