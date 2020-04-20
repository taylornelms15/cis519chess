import logging
import enum
import numpy as np
import re


class Occupier(enum.IntEnum):
    WHITE = 1
    CLEAR = 0
    BLACK = -1

class PieceType(enum.IntEnum):
    PAWN       = 0
    ROOK       = 1
    KNIGHT     = 2
    BISHOP     = 3
    QUEEN      = 4
    KING       = 5

#First letter is "white" label in FEN, second is "black"
PIECELABELS = {PieceType.PAWN:      ["P", "p"],
               PieceType.ROOK:      ['R', 'r'],
               PieceType.KNIGHT:    ['N', 'n'],
               PieceType.BISHOP:    ['B', 'b'],
               PieceType.QUEEN:     ['Q', 'q'],
               PieceType.KING:      ['K', 'k']}

FENParseString = "(\w+)/(\w+)/(\w+)/(\w+)/(\w+)/(\w+)/(\w+)/(\w+)\s*([w|b]*)\s*([KQkq]*)"  

class BitBoard():
    """
    Class representing bitmap for a single piece type
    Filled with WHITE, CLEAR, or BLACK values
    """
    __slots__ = ["board", "pieceType"]
    def __init__(self, pieceType, fenString = None):
        self.board = np.zeros((8,8)).astype(np.int8)
        self.pieceType = pieceType

        if fenString:
            self.constructFromFen(fenString)
        
    def constructFromFen(self, fenString):
        """
        Makes bitboard match the given fen string
        assumes any capital letter is white, lowercase is black
        """
        #TODO: WAAAAAAY more error checking
        regstr = FENParseString
        fenMatches = re.match(regstr, fenString).groups()

        for i in range(8):
            thisRowFen = fenMatches[i]
            pos = 0
            for char in thisRowFen:
                if re.match("\d", char):#if numeric
                    numSpaces = int(char)
                    for j in range(numSpaces):
                        self.board[i][pos] = Occupier.CLEAR
                        pos += 1
                elif re.match("[A-Z]", char):#if capital
                    self.board[i][pos] = Occupier.WHITE
                    pos += 1
                elif re.match("[a-z]", char):#if lowercase
                    self.board[i][pos] = Occupier.BLACK
                    pos += 1
                else:
                    raise ValueError("Invalid character %s in fen string" % char)



    def asFen(self):
        """
        Quick and dirty way to get a FEN-style representation of this structure
        """
        retval = ""
        for i in range(self.board.shape[0]):
            runningTotal = 0
            for j in range(self.board.shape[1]):
                val = self.board[i][j]
                if (val == Occupier.CLEAR):
                    runningTotal += 1
                elif (val == Occupier.WHITE):
                    if runningTotal > 0:
                        retval += str(runningTotal)
                        runningTotal = 0
                    retval += PIECELABELS[self.pieceType][0]
                else:
                    if runningTotal > 0:
                        retval += str(runningTotal)
                        runningTotal = 0
                    retval += PIECELABELS[self.pieceType][1]

            if runningTotal > 0:
                retval += str(runningTotal)
            if i != self.board.shape[0] - 1:
                retval += '/'

        return retval

    def __str__(self):
        """
        Note: Board may look flipped due to numpy array indexing
        """
        retval = "<BitBoard %s\n%s>" % (self.pieceType.name, self.board)
        return retval

