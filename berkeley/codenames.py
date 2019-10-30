import random
import numpy as np

COLORS = {'Blue': 'Blue', 'Red': 'Red'}


def getColorMap(board, firstColor=COLORS['Blue']):
    print(firstColor, 'goes first!')
    colorMap = {}
    indeces = random.sample(range(25), 18)
    color = 'Blue'
    for _ in range(2):
        for _ in range(8):
            colorMap[indeces.pop()] = color
        color = 'Red'
    colorMap[indeces.pop()] = firstColor
    colorMap[indeces.pop()] = 'Black'
    assert len(indeces) == 0
    return colorMap


def getWordsByColor(board, colorMap, color):
    board_list = board.reshape(25)
    return [board_list[index]
            for index in colorMap.keys() if colorMap[index] == color]


def getRandomBoard(wordlist):
    words = np.array(random.sample(wordlist, 25))
    board = words.reshape((5, 5))
    return board


def getLongestWordLength(board):
    longest = 0
    for row in board:
        for word in row:
            if len(word) > longest:
                longest = len(word)
    return longest


def printGrid(grid, longest):
    for row in grid:
        for word in row:
            print('{0:<{width}}'.format(word, width=longest + 1), end='')
        print()


if __name__ == '__main__':
    wordlist = []
    with open('wordlist.txt', 'r') as f:
        for word in f:
            wordlist.append(word.strip())
    board = getRandomBoard(wordlist)
    longest = getLongestWordLength(board)
    printGrid(board, longest)
    colorMap = getColorMap(board, random.choice(list(COLORS.keys())))
    print(colorMap)
    blue_words = getWordsByColor(board, colorMap, 'Blue')
    red_words = getWordsByColor(board, colorMap, 'Red')
    black_words = getWordsByColor(board, colorMap, 'Black')
    print(blue_words)
    print(red_words)
    print(black_words)
