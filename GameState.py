import logging
import enum
import re

from log import setupLogging
from BitBoard import BitBoard, PieceType, BitBoardsFromFenString, FENParseString

import pdb



class Turn(enum.IntEnum):
    WHITE = 0
    BLACK = 1
   
class Castle(enum.IntEnum):
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
    __slots__ = ['bitboards', 'turn', 'possibleCastles', 'halfmoveClock']
    initialBoardFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def __init__(self, startingFrom = None, fenString = None):
        if startingFrom:
            self.bitboards = [x for x in startingFrom.bitboards]#should do a copy of these, rather than by-reference
            if startingFrom.turn == Turn.WHITE:
                self.turn = Turn.BLACK
            else:
                self.turn = Turn.WHITE
            self.possibleCastles = [x for x in startingFrom.possibleCastles]
            self.halfmoveClock = startingFrom.halfmoveClock
        elif fenString:#If starting from a fenstring representation of a game state
            self.constructFromFenString(fenString)
        else:
            self.bitboards = [] #bitfields representing piece positions
            self.turn = Turn.WHITE #whose turn it is
            self.possibleCastles = [True, True, True, True]#whether any of the four castle types can be done (only cares about whether the relevant pieces have moved previously, not the other castling rules)
            self.halfmoveClock = 0#number of half-moves (turns) since last pawn move or piece caputre, for determining draw (50-turn rule)

    def constructFromFenString(self, fenString):
        """
        Makes the game state from a given FEN string
        """
        self.bitboards = BitBoardsFromFenString(fenString)
        fenPieces = re.match(FENParseString, fenString).groups()
        turnString = fenPieces[8]
        castleString = fenPieces[9]
        if turnString == "b":
            self.turn = Turn.BLACK
        else:
            self.turn = Turn.WHITE
        self.possibleCastles = self.possCastlesFromCastleFen(castleString)

    def possCastlesFromCastleFen(self, castleString):
        retval = [False, False, False, False]
        if re.match("K", castleString) != None:
            retval[Castle.WKING] = True
        if re.match("Q", castleString) != None:
            retval[Castle.WQUEEN] = True
        if re.match("k", castleString) != None:
            retval[Castle.BKING] = True
        if re.match("q", castleString) != None:
            retval[Castle.BQUEEN] = True
        return retval

    @classmethod
    def getInitialState(cls):
        """
        Class constructor that creates the state of the initial chess board
        """
        return GameState(fenString = cls.initialBoardFen)    

    # Functions to treat a GameState more atomically in dictionaries and whatnot

    def __repr__(self):
        retval = "<GameState %s>" % list(self._key())
        return retval

    def _key(self):
        return (self.bitboards, self.turn, self.possibleCastles)#, self.halfmoveClock)

    def __hash__(self):
        return hash(self._key())

    def __eq__(self, other):
        if isinstance(other, GameState):
            return self._key() == other._key()
        return NotImplemented

    # Accessors

    def getBitboards(self):
        return self.bitboards

    def getTurn(self):
        return self.turn

    def getPossibleCastles(self):
        return self.possibleCastles

    def getHalfmoveClock(self):
        return self.halfmoveClock

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

    def isCheck(self):
        """
        Booleanfor whether this game state has the current player's king in check
        """
        raise NotImplementedError

class Move(object):
    __slots__ = ["startLoc", "endLoc", "castle", "promotion"]

    def __init__(self, startLoc, endLoc, castle = None, promotion = None):
        self.startLoc   = startLoc
        self.endLoc     = endLoc
        self.castle     = castle
        self.promotion  = promotion


    def apply(self, gameState):
        """
        Applies the move of the piece from startLoc to endLoc
        Does not check if the move is legal

        @return The GameState after this move is applied
        """
        raise NotImplementedError

    @classmethod
    def constructCastle(cls, castleType):
        return Move(None, None, castleType, None)

    @classmethod
    def constructFromPgnHalfmove(cls, gameState, Piece, Rank, File, Endloc, Promotion):


        pdb.set_trace()
        raise NotImplementedError
  















def main():
    myState = GameState.getInitialState()
    logging.info(myState)



if __name__ == "__main__":
    setupLogging()
    main()

