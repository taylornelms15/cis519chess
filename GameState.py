import logging
import enum
from log import setupLogging

class Turn(enum.Enum):
    WHITE = 0
    BLACK = 1

class GameState(object):

    def __init__(self):
        self.bitboards = [] #bitfields representing piece positions
        self.turn = Turn.WHITE #whose turn it is
        self.castleWKPoss = True #if can still do a white kingside castle (only cares about whether those pieces have been moved)
        self.castleWQPoss = True #if can still do a white queenside castle
        self.castleBKPoss = True #if can still do a black kingside castle
        self.castleBQPoss = True #if can still do a black queenside castle


















def main():
    myState = GameState()
    logging.info(myState)
    pass



if __name__ == "__main__":
    setupLogging()
    main()

