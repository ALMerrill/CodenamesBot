import codenames
import random
import time
import io
import numpy as np
from scipy import spatial

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

class BaselineHintGiver:
    def __init__(self, wordlist):
        self.data = load_vectors('/home/bandrus/applications/word_embeddings/fastText/fil9.vec')

        with open('top_50k_cleaned.txt') as in_file:
            vocabulary = in_file.readlines()
        self.vocabulary = [word.strip().lower() for word in vocabulary if word.strip().lower() not in wordlist]
        self.cards = wordlist
        self.similarities = np.zeros((len(vocabulary), len(wordlist)))
        self.vocab_vectors = {vocab_word: np.fromiter(self.data[vocab_word], dtype=float) for vocab_word in self.vocabulary}
        self.card_vectors = {card_word: np.fromiter(self.data[card_word], dtype=float) for card_word in wordlist}
        for i, vocab_word in enumerate(self.vocabulary):
            for j, card_word in enumerate(wordlist):
                vocab_vector = self.vocab_vectors[vocab_word]
                card_vector = self.card_vectors[card_word]
                self.similarities[i][j] = np.dot(vocab_vector, card_vector)
        self.std_similarity = np.std(self.similarities)
        print('Calculated all similarities')


    # Not actually good, just wanted to see how well it would behave. Turns out not very well.
    def give_hint_basic(self, my_indices, bad_indices, assassin_index):
        word_scores = []
        for row in self.similarities:
            score = 0
            for index in my_indices:
                score += row[index] ** 2
            for index in bad_indices:
                score -= row[index] ** 2
            score -= 10 * row[assassin_index] ** 2
            word_scores.append(score)
        return self.vocabulary[np.argmax(word_scores)]

    def give_hint2(self, my_indices, bad_indices, assassin_index):
        best_hint = ''
        best_number = 0
        for i, row in enumerate(self.similarities):
            good_options = [row[index] for index in my_indices]
            bad_options = [row[index] for index in bad_indices]
            worst_option = row[assassin_index]

            good_options.sort(reverse=True)
            bad_options.sort(reverse=True)

            benchmark = max(bad_options[0] + self.std_similarity, worst_option + 2 * self.std_similarity)

            num = len([option for option in good_options if option > benchmark])

            if num > best_number:
                best_number = num
                best_hint = self.vocabulary[i]
                print(best_hint, best_number)

        return best_hint, best_number

    def give_hint(self, my_indices, bad_indices, assassin_index):
        best_hint = ''
        best_number = 0
        for i, row in enumerate(self.similarities):
            closest_words = np.argsort(-row)
            for j, close_word in enumerate(closest_words):
                if close_word not in my_indices:
                    break
            if j > best_number:
                best_number = j
                best_hint = self.vocabulary[i]
                print(best_hint, best_number)
        return best_hint, best_number


    def give_hint3(self, my_indices, bad_indices, assassin_index):
        best_hint = ''
        best_number = 0
        for i, row in enumerate(self.similarities):
            probs = softmax(row)

            bad_prob = max([probs[ind] for ind in bad_indices])
            really_bad_prob = probs[assassin_index]

            good_probs = [(ind, probs[ind]) for ind in my_indices]

            good_hints = len([0 for ind, prob in good_probs if prob >= bad_prob * 2 and prob >= really_bad_prob * 3])

            if good_hints >= 2:
                print(good_hints)

            # two_bar = 0.4
            # if np.count_nonzero(probs > two_bar) >= 2:
            #     print(2, self.vocabulary[i], [self.cards[ind] for ind, prob in enumerate(probs) if prob > two_bar], [prob for prob in probs if prob > two_bar])
            #
            # three_bar = 0.20
            # if np.count_nonzero(probs > three_bar) >= 3:
            #     print(3, self.vocabulary[i], [self.cards[ind] for ind, prob in enumerate(probs) if prob > three_bar], [prob for prob in probs if prob > three_bar])

            # for j, close_word in enumerate(closest_words):
            #     if close_word not in my_indices:
            #         break
            # if j > best_number:
            #     best_number = j
            #     best_hint = self.vocabulary[i]
            #     print(best_hint, best_number)
        return best_hint, best_number


def main():
    start = time.time()
    with open('shortened_wordlist.txt') as in_file:
      wordlist = in_file.readlines()
    wordlist = [word.strip().lower() for word in wordlist]

    board = codenames.getRandomBoard(wordlist)
    board = board.reshape((25))
    colormap = codenames.getColorMap(board, random.choice(['Blue', 'Red']))
    blue_indices = [index for index, color in colormap.items() if color == 'Blue']
    red_indices = [index for index, color in colormap.items() if color == 'Red']
    assassin_index = [index for index, color in colormap.items() if color == 'Black'][0]

    hintGiver = BaselineHintGiver(board)

    hintGiver.give_hint3(blue_indices, red_indices, assassin_index)
    # hint = hintGiver.give_hint(blue_indices, red_indices, assassin_index)
    # hint2 = hintGiver.give_hint2(blue_indices, red_indices, assassin_index)
    #
    # print('Want to guess:', [board[index] for index in blue_indices])
    # print('Don\'t want to guess:', [board[index] for index in red_indices])
    # print('Definitely don\'t want to guess:', board[assassin_index])
    # print('Hint is', hint)
    # print('Other hint is', hint2)
    #
    #
    # end = time.time()
    # print(f'Took {end - start} seconds')

if __name__== "__main__":
    main()
