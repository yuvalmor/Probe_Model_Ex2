import sys
from collections import Counter

from lidstone import lidstone


def is_header(text):
    return text.strip().startswith("<") and text.strip().endswith(">")


def parse_file(file_name):
    with open(file_name) as f:
        # read and parse setring. avoid headers
        content = list(filter(lambda a: not is_header(a), f.readlines()))
        content = [line.strip() for line in content]  # remove \n char
    content = [line.split() for line in content]  # split by white space
    content = [
        item for sublist in content for item in sublist
    ]  # put all words in flat list
    return content


def float_range(start, end, step):
    current = start
    while current <= end:
        yield round(current, 2)
        current += step


voc_size = 300000  # given
""" init - part 1 """
print("Output1:", sys.argv[1], sep="\t")
print("Output2:", sys.argv[2], sep="\t")
print("Output3:", sys.argv[3], sep="\t")
print("Output4:", sys.argv[4], sep="\t")
print("Output5:", voc_size, sep="\t")
print("Output6:", 1 / voc_size, sep="\t")  # uniform distribution

""" Development set preprocessing - part 2 """
content = parse_file(sys.argv[1])  # parse dev file
number_of_words = len(content)
print("Output7:", number_of_words, sep="\t")

""" Lidstone model training - part 3 """
training, vald = (
    content[: round(0.9 * len(content))],
    content[round(0.9 * len(content)) :],
)
print("Output8:", len(vald), sep="\t")  # number of events in the validation set
print("Output9:", len(training), sep="\t")  # number of event in the training set
training_counter = Counter(training)  # count words in the training set
print(
    "Output10:", len(training_counter), sep="\t"
)  # different events in the trainign set
print(
    "Output11:", training_counter[sys.argv[3]], sep="\t"
)  # number of times of input word


lidstone_models = {}
for lambda_val in float_range(0, 2, 0.01):
    model = lidstone(training_counter, training, lambda_val)
    perp = model.get_perplexity(vald)
    lidstone_models[lambda_val] = {"model": model, "perplexity": perp}


print("Output12:", lidstone_models[0]["model"].get_prob(sys.argv[3]), sep="\t")
print("Output13:", lidstone_models[0]["model"].get_prob("unseen-word"), sep="\t")
print("Output14:", lidstone_models[0.10]["model"].get_prob(sys.argv[3]), sep="\t")
print("Output15:", lidstone_models[0.10]["model"].get_prob("unseen-word"), sep="\t")
print("Output16:", lidstone_models[0.01]["perplexity"], sep="\t")
print("Output17:", lidstone_models[0.10]["perplexity"], sep="\t")
print("Output18:", lidstone_models[1.00]["perplexity"], sep="\t")
best_lambda = 0
for lam_val in lidstone_models:
    if (
        lidstone_models[lam_val]["perplexity"]
        < lidstone_models[best_lambda]["perplexity"]
    ):
        best_lambda = lam_val
print("Output19:", best_lambda, sep="\t")
print("Output19:", lidstone_models[best_lambda]["perplexity"], sep="\t")
