""" Yuval mor, Roey Fuchs 205380173, 205415342 """
from utils import perplexity, voc_size


class lidstone:
    def __init__(self, counter_words, training_set, lambda_value):
        self.counter_words = counter_words
        self.training_set = training_set
        self.lambda_value = lambda_value

    def get_prob(self, word):
        """ return the probability of input word """
        return self.get_prob_by_word_freq(self.counter_words[word])

    def get_prob_by_word_freq(self, r):
        """ return the probability of a word with input frequency using the
        lambda if the spesific model as input in the constructor """
        return (r + self.lambda_value) / (
            len(self.training_set) + self.lambda_value * voc_size
        )

    def get_est_freq_by_freq(self, r):
        """ return expected frequency by input frequency """
        return self.get_prob_by_word_freq(r) * len(self.training_set)

    def get_perplexity(self, data):
        """ return perplexity of the model with input data """
        return perplexity(self, data)
