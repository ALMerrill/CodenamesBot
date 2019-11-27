import codenames
from hint_giver import BaselineHintGiver
from guesser import BaselineGuesser
import random

def main():
    with open('shortened_wordlist.txt') as in_file:
      wordlist = in_file.readlines()
    wordlist = [word.strip().lower() for word in wordlist]

    board = codenames.getRandomBoard(wordlist)
    board = board.reshape((25))
    og_board = board[:] # Shallow copy for posterity's sake
    colormap = codenames.getColorMap(board, random.choice(['Blue', 'Red']))
    team_indices = {}
    team_indices['Blue'] = [index for index, color in colormap.items() if color == 'Blue']
    team_indices['Red'] = [index for index, color in colormap.items() if color == 'Red']
    assassin_index = [index for index, color in colormap.items() if color == 'Black'][0]

    hint_giver = BaselineHintGiver(board)
    guesser = BaselineGuesser()

    whose_turn = 'Blue' if len(team_indices['Blue']) > len(team_indices['Red']) else 'Red'

    others_team = 'Blue' if whose_turn == 'Red' else 'Red'
    print(f'Starting with {whose_turn}s turn and not {others_team}s turn!')

    print('Red indices:', team_indices['Red'])
    print('Blue indices:', team_indices['Blue'])


    # for i, word in enumerate(og_board):
    #     print(i, word)

    game_over = False
    while not game_over:
        hint, num = hint_giver.give_hint3(team_indices[whose_turn], team_indices[others_team], assassin_index)

        print(f'Hint for {whose_turn} is {hint}: {num}')

        guesses = guesser.guess(board, hint, num)

        print(guesses)

        for guess in guesses:
            guess_ind = og_board.tolist().index(guess)
            print(f'Guess {guess} is at index {guess_ind}')
            if guess_ind in team_indices[whose_turn]:
                print('Got it right!')
            elif guess_ind == assassin_index:
                print('Died instantly!')
            elif guess_ind in team_indices[others_team]:
                print('Helped your enemy')
            else:
                print('Got it wrong.')

        # Adjust, decide if game should be over

        # whose_turn, others_team = others_team, whose_turn

        break




if __name__== "__main__":
    main()
