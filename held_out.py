from collections import Counter, defaultdict

from utils import perplexity, voc_size


class held_out:
    def __init__(self, training_set):
        self.train = training_set[: int(len(training_set) / 2)]
        self.held = training_set[int(len(training_set) / 2) :]
        self.r = Counter(self.train)
        self.Nr = defaultdict(int)
        for val in self.r.values():
            self.Nr[val] += 1
        self.Nr[0] = voc_size - len(self.r)
        self.ch = Counter(self.held)
        self.tr = defaultdict(int)

        for key, val in self.r.items():
            self.tr[val] += self.ch[key]

        for word, count in self.ch.items():
            if self.r[word] == 0:
                self.tr[0] += count

    def get_prob(self, word):
        word_r = self.r[word]
        return self.tr[word_r] / (self.Nr[word_r] * len(self.held))

    def get_perplexity(self, data):
        return perplexity(self, data)
