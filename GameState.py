import logging
import enum
from log import setupLogging

class Turn(enum.Enum):
    WHITE = 0
    BLACK = 1
   
class Castle(enum.Enum):
    """ 
    Four different castle moves:
        White Kingside
        White Queenside
        Black Kingside
        Black Queenside
    Useful for indexing into arrays, probably
    """
    WKING   = 0
    WQUEEN  = 1
    BKING   = 2
    BQUEEN  = 3

class GameState(object):

    def __init__(self, startingFrom = None):
        if startingFrom:
            self.bitboards = [x for x in startingFrom.bitboards]#should do a copy of these, rather than by-reference
            if startingFrom.turn == Turn.WHITE:
                self.turn = Turn.BLACK
            else:
                self.turn = Turn.WHITE
            self.possibleCastles = [x for x in startingFrom.possibleCastles]

        else:
            self.bitboards = [] #bitfields representing piece positions
            self.turn = Turn.WHITE #whose turn it is
            self.possibleCastles = [True, True, True, True]#whether any of the four castle types can be done (only cares about whether the relevant pieces have moved previously, not the other castling rules)

    # Functions to treat a GameState more atomically in dictionaries and whatnot

    def _key(self):
        return (self.bitboards, self.turn, self.possileCastles)

    def __hash__(self):
        return hash(self._key())

    def __eq__(self, other):
        if isinstance(other, GameState):
            return self._key() == other._key()
        return NotImplemented

    # Chess-specific functions

    def getPossibleMoves(self):
        """
        Given the current game state, returns a list of all possible moves that could be performed
        """
        raise NotImplementedError

    def isCheckmate(self):
        """
        Boolean for whether this game state is a checkmate for the player whose turn it is
        """
        raise NotImplementedError



class Move(object):

    def __init__(self, startLoc, endLoc):
        self.startLoc   = startLoc
        self.endLoc     = endLoc


    def apply(self, gameState):
        """
        Applies the move of the piece from startLoc to endLoc
        Does not check if the move is legal

        @return The GameState after this move is applied
        """
        raise NotImplementedError
  
















def main():
    myState = GameState()
    logging.info(myState)
    pass



if __name__ == "__main__":
    setupLogging()
    main()

