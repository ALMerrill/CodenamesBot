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


class BaselineGuesser:
    def __init__(self):
        self.data = load_vectors('/home/bandrus/applications/word_embeddings/fastText/fil9.vec')

        with open('top_50k_cleaned.txt') as in_file:
            vocabulary = in_file.readlines()
        self.vocabulary = [word.strip().lower() for word in vocabulary]
        self.vocab_vectors = {vocab_word: np.fromiter(self.data[vocab_word], dtype=float) for vocab_word in self.vocabulary}



    def guess(self, options, hint, num):
        option_vectors = [self.vocab_vectors[option.lower()] if option else None for option in options]
        hint_vector = self.vocab_vectors[hint.lower()]

        similarities = [np.dot(hint_vector, option_vector) if option_vector is not None else 0 for option_vector in option_vectors]

        guesses = []
        for i in range(num):
            guess_index = similarities.index(max(similarities))
            guesses.append(options[guess_index])
            similarities[guess_index] = 0
        return guesses


def main():
  guesser = BaselineGuesser()
  cards = ['Queen', 'Well', 'Torch', 'Vacuum', 'Snow', 'Turkey', 'Helicopter',
            'Eagle', 'Plastic', 'Bark','Cricket', 'Hook', 'Press', 'Alien',
            'Pan', 'Dinosaur', 'Nurse', 'Apple', 'Kid', 'Fly', 'Moscow']

  hint = 'Bird'
  hint_num = 3
  guesses = guesser.guess(cards, hint, hint_num)
  print(guesses)

if __name__== "__main__":
  main()
