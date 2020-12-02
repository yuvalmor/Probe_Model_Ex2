import sys
from collections import Counter

from held_out import held_out
from lidstone import lidstone
from utils import sum_model_probs, voc_size

DEBUG = False  # for part 5


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


f = open(sys.argv[4], "w")
print(
    "#Students", "Yuval Mor", "Roey Fuchs", "205380173", "205415342", sep="\t", file=f
)
""" init - part 1 """
print("#Output1", sys.argv[1], sep="\t", file=f)
print("#Output2", sys.argv[2], sep="\t", file=f)
print("#Output3", sys.argv[3], sep="\t", file=f)
print("#Output4", sys.argv[4], sep="\t", file=f)
print("#Output5", voc_size, sep="\t", file=f)
print("#Output6", 1 / voc_size, sep="\t", file=f)  # uniform distribution

""" Development set preprocessing - part 2 """
content = parse_file(sys.argv[1])  # parse dev file
number_of_words = len(content)
print("#Output7", number_of_words, sep="\t", file=f)

""" Lidstone model training - part 3 """
training, vald = (
    content[: round(0.9 * len(content))],
    content[round(0.9 * len(content)) :],
)
print("#Output8", len(vald), sep="\t", file=f)  # number of events in the validation set
print(
    "#Output9", len(training), sep="\t", file=f
)  # number of event in the training set
training_counter = Counter(training)  # count words in the training set
print(
    "#Output10", len(training_counter), sep="\t", file=f
)  # different events in the trainign set
print(
    "#Output11", training_counter[sys.argv[3]], sep="\t", file=f
)  # number of times of input word


lidstone_models = {}
for lambda_val in float_range(0, 2, 0.01):
    model = lidstone(training_counter, training, lambda_val)
    perp = model.get_perplexity(vald)
    lidstone_models[lambda_val] = {"model": model, "perplexity": perp}


print("#Output12", lidstone_models[0]["model"].get_prob(sys.argv[3]), sep="\t", file=f)
print(
    "#Output13", lidstone_models[0]["model"].get_prob("unseen-word"), sep="\t", file=f
)
print(
    "#Output14", lidstone_models[0.10]["model"].get_prob(sys.argv[3]), sep="\t", file=f
)
print(
    "#Output15",
    lidstone_models[0.10]["model"].get_prob("unseen-word"),
    sep="\t",
    file=f,
)
print("#Output16", lidstone_models[0.01]["perplexity"], sep="\t", file=f)
print("#Output17", lidstone_models[0.10]["perplexity"], sep="\t", file=f)
print("#Output18", lidstone_models[1.00]["perplexity"], sep="\t", file=f)
best_lambda = 0
for lam_val in lidstone_models:
    if (
        lidstone_models[lam_val]["perplexity"]
        < lidstone_models[best_lambda]["perplexity"]
    ):
        best_lambda = lam_val
print("#Output19", best_lambda, sep="\t", file=f)
print("#Output20", lidstone_models[best_lambda]["perplexity"], sep="\t", file=f)

held = held_out(content)
print("#Output21", len(held.train), sep="\t", file=f)
print("#Output22", len(held.held), sep="\t", file=f)
print("#Output23", held.get_prob(sys.argv[3]), sep="\t", file=f)
print("#Output24", held.get_prob("unseen-word"), sep="\t", file=f)

""" debug code - part 5 """
if DEBUG:
    print(
        "DEBUG LIDSTONE",
        sum_model_probs(
            lidstone_models[best_lambda]["model"],
            lidstone_models[best_lambda]["model"].counter_words,
        ),
        file=f,
    )
    print("DEBUG HELD OUT", sum_model_probs(held, held.r), file=f)
""" test - part 6 """
content_test = parse_file(sys.argv[2])  # parse test file
print("#Output25", len(content_test), sep="\t", file=f)
print(
    "#Output26",
    lidstone_models[best_lambda]["model"].get_perplexity(content_test),
    sep="\t",
    file=f,
)
print("#Output27", held.get_perplexity(content_test), sep="\t", file=f)

print(
    "#Output28",
    "H"
    if held.get_perplexity(content_test)
    < lidstone_models[best_lambda]["model"].get_perplexity(content_test)
    else "L",
    sep="\t",
    file=f,
)
for i in range(10):
    print(
        i,
        round(lidstone_models[best_lambda]["model"].get_est_freq_by_freq(i), 5),
        round(held.get_est_freq_by_freq(i), 5),
        round(held.Nr[i], 5),
        round(held.tr[i], 5),
        sep="\t",
        file=f,
    )
f.close()
