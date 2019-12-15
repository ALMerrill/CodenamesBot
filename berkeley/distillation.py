import signal
import io
from os import path
import numpy as np
import torch
import torch.optim as optim

new_size = 80
new_vectors_path = f'new-word-vectors/distillation-{new_size}.vec'
learning_rate = 0.001
learning_decay = 0.99

OVERFITTING = True
if OVERFITTING:
    new_vectors_path = f'new-word-vectors/overfit-distillation-{new_size}.vec'

with open('shortened_wordlist.txt') as main_word_file:
    main_words = [word.lower().strip() for word in main_word_file.readlines()]

# if OVERFITTING:
#     main_words = main_words[:25]

new_word_vectors = {}

def sigint_handler(sig, frame):
    global new_word_vectors
    print('Writing current vectors to file...')
    with open(new_vectors_path, 'w') as out_file:
        print(f'50000 {new_size}', file=out_file)
        for key, val in zip(new_word_vectors.keys(), new_word_vectors.values()):
            print(key, end=' ', file=out_file)
            for dim in val.tolist():
                print(dim, end=' ', file=out_file)
            print(file=out_file)
    print('Done')
    exit()

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

def rand_init_vectors(vocabulary):
    init_vectors = {}
    for word in vocabulary:
        init_vectors[word] = torch.from_numpy(np.random.normal(0, 0.2, new_size))
    return init_vectors


def report_relevant_progress(canonical_vectors, wip_vectors, vocabulary):
    test_words = np.random.choice(vocabulary, 10, replace=False)

    error = 0
    for word in main_words:
        real_similarities = [torch.dot(canonical_vectors[word], canonical_vectors[test_word]) for test_word in test_words]
        wip_similarities = [torch.dot(wip_vectors[word], wip_vectors[test_word]) for test_word in test_words]

        real_order = np.argsort(real_similarities)
        wip_order = np.argsort(wip_similarities)

        for rov, wov in zip(real_order, wip_order):
            error += abs(rov - wov)
    print(f'Error is {error}')

def train(canonical_vectors, wip_vectors, vocabulary):
    print('Training...')
    global new_word_vectors
    new_word_vectors = wip_vectors
    iters = 0
    while True:
        iters += 1
        words = np.random.choice(vocabulary, 3, replace=False)
        real_sim_1 = np.dot(canonical_vectors[words[0]], canonical_vectors[words[1]])
        real_sim_2 = np.dot(canonical_vectors[words[0]], canonical_vectors[words[2]])
        predicted_sim_1 = np.dot(new_word_vectors[words[0]], new_word_vectors[words[1]])
        predicted_sim_2 = np.dot(new_word_vectors[words[0]], new_word_vectors[words[2]])

        real_diff = real_sim_1 - real_sim_2  # How much closer words[1] is to words[0] than words[2] is to words[0]
        predicted_diff = predicted_sim_1 - predicted_sim_2

        # If real_diff is positive and predicted diff is negative, words[2] needs to move away from words[0]
            # This is when loss is super positive
        # If real_diff is negative and predicted diff is positive, words[1] needs to move away from words[0]
            # This is when loss is super negative

        loss = real_diff - predicted_diff

        step = loss * learning_rate

        # If step is postivie, move words[1] closer and words[2] away
        # If step is negative, move words[2] closer and words[1] away

        for i in range(new_size):
            dim_step = step * np.sign(new_word_vectors[words[0]])

            new_word_vectors[words[1]] += dim_step
            new_word_vectors[words[2]] -= dim_step

        if iters % 5000 == 0:
            print(iters, end=' ')
            report_progress(canonical_vectors, new_word_vectors, vocabulary)


def train_overfit(canonical_vectors, wip_vectors, vocabulary):
    print('Training...')
    global new_word_vectors
    global learning_rate
    global learning_decay
    new_word_vectors = wip_vectors
    iters = 0
    losses = []
    while True:
        iters += 1

        for test_word in vocabulary:
        # key_word = np.random.choice(main_words, 1)[0]
        # test_word = np.random.choice(vocabulary, 1)[0]

        # if test_word in main_words:
        #     continue

        # import pdb; pdb.set_trace()
            loss = 0
            optimizer = optim.Adam([new_word_vectors[test_word].requires_grad_()],lr=learning_rate)
            for word in main_words:
                real_sim = torch.dot(canonical_vectors[word], canonical_vectors[test_word])
                predicted_sim = torch.dot(new_word_vectors[word], new_word_vectors[test_word])
                loss += torch.abs(real_sim - predicted_sim)

            losses.append(loss.item())

            loss.backward()
            optimizer.step()

        learning_rate *= learning_decay
        print(iters, end=' ')
        print(learning_rate, end=' ')
        print(np.mean(losses))
        losses = []
                # report_relevant_progress(canonical_vectors, new_word_vectors, vocabulary)



def main():
    signal.signal(signal.SIGINT, sigint_handler)

    canonical_data = load_vectors('/home/bandrus/applications/word_embeddings/fastText/fil9.vec')

    with open('top_50k_cleaned.txt') as in_file:
        vocabulary = in_file.readlines()
    vocabulary = [word.strip().lower() for word in vocabulary]
    if OVERFITTING:
        vocabulary = list(set(vocabulary[:200] + main_words)) # 200 works great. 1000 bottomed out around loss==1
    canonical_vectors = {vocab_word: torch.from_numpy(np.fromiter(canonical_data[vocab_word], dtype=float)) for vocab_word in vocabulary}

    if path.exists(new_vectors_path):
        wip_vectors_data = load_vectors(new_vectors_path)
        wip_vectors = {vocab_word: torch.from_numpy(np.fromiter(wip_vectors_data[vocab_word], dtype=float)) for vocab_word in vocabulary}
    else:
        wip_vectors = rand_init_vectors(vocabulary)
        # if new_size == 100:
        #     for word in main_words:
        #         wip_vectors[word] = canonical_vectors[word].clone().detach()


    train_overfit(canonical_vectors, wip_vectors, vocabulary)


if __name__== "__main__":
    main()
