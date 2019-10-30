import codenames
import random
import time
import io
import numpy as np

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

class BaselineGuessGiver:
    def __init__(self, wordlist, my_indices, bad_indices, assassin_index):
        self.data = load_vectors('/home/bandrus/applications/word_embeddings/fastText/fil9.vec')

        with open('top_50k_cleaned.txt') as in_file:
            vocabulary = in_file.readlines()
        vocabulary = [word.strip().lower() for word in vocabulary if word.strip().lower() not in wordlist]
        self.similarities = np.zeros((len(vocabulary), len(wordlist)))
        self.vocab_vectors = {vocab_word: np.fromiter(self.data[vocab_word], dtype=float) for vocab_word in vocabulary}
        self.card_vectors = {card_word: np.fromiter(self.data[card_word], dtype=float) for card_word in wordlist}
        for i, vocab_word in enumerate(vocabulary):
            for j, card_word in enumerate(wordlist):
                vocab_vector = self.vocab_vectors[vocab_word]
                card_vector = self.card_vectors[card_word]
                self.similarities[i][j] = np.dot(vocab_vector, card_vector)
        print('Calculated all similarities')



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

    guessGiver = BaselineGuessGiver(board, blue_indices, red_indices, assassin_index)

    end = time.time()
    print(f'Took {end - start} seconds')

if __name__== "__main__":
    main()
