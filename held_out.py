""" Yuval mor, Roey Fuchs 205380173, 205415342 """
from collections import Counter, defaultdict

from utils import perplexity, voc_size


class held_out:
    def __init__(self, training_set):
        # split set to train and held out sets
        self.train = training_set[: int(len(training_set) / 2)]
        self.held = training_set[int(len(training_set) / 2) :]
        # count words freq
        self.r = Counter(self.train)
        # default dict create a dict that keys are ints and their values
        # are 0
        self.Nr = defaultdict(int)
        for val in self.r.values():
            self.Nr[val] += 1
        # calculate how many words don't appear in the train set
        self.Nr[0] = voc_size - len(self.r)
        self.ch = Counter(self.held)
        self.tr = defaultdict(int)

        for key, val in self.r.items():
            self.tr[val] += self.ch[key]

        for word, count in self.ch.items():
            if self.r[word] == 0:
                self.tr[0] += count

    def get_prob(self, word):
        """ will return the probability of input word """
        return self.get_prob_by_word_freq(self.r[word])

    def get_prob_by_word_freq(self, r):
        """ return the probability of a word with input frequency """
        return self.tr[r] / (self.Nr[r] * len(self.held))

    def get_est_freq_by_freq(self, r):
        """ return the expected frequency of word with input frequency """
        return self.get_prob_by_word_freq(r) * len(self.train)

    def get_perplexity(self, data):
        """ return perplexity of the model with input data """
        return perplexity(self, data)
