import logging
import enum
import numpy as np
import re


class Occupier(enum.IntEnum):
    WHITE = 1
    CLEAR = 0
    BLACK = -1

class PieceType(enum.IntEnum):
    PAWN        = 0
    ROOK        = 1
    KNIGHT      = 2
    BISHOP      = 3
    QUEEN       = 4
    KING        = 5

def S2I(posString):
    """
    (String 2 Index)
    Turns a string representing a position to an index that
    can be used to index into a bitboard
    """
    return ((ord(posString[1]) - ord('1')), (ord(posString[0]) - ord('a')))

def I2S(posTuple):
    """
    (Index 2 String)
    Reverse of the above
    """
    return "%s%s" % (chr(posTuple[1] + ord('a')), chr(posTuple[0] + ord('0')))

#First letter is "white" label in FEN, second is "black"
PIECELABELS = {PieceType.PAWN:      ["P", "p"],
               PieceType.ROOK:      ['R', 'r'],
               PieceType.KNIGHT:    ['N', 'n'],
               PieceType.BISHOP:    ['B', 'b'],
               PieceType.QUEEN:     ['Q', 'q'],
               PieceType.KING:      ['K', 'k']}

FENParseString = "(\w+)/(\w+)/(\w+)/(\w+)/(\w+)/(\w+)/(\w+)/(\w+)\s*([w|b]*)\s*([KQkq-]*)\s*([-\w]+)\s*"  

def BitBoardsFromFenString(fenString):
    """
    From a single FEN string, creates a set of 6 bit boards that represent that board state
    """
    retval = [None] * len(PieceType)

    firstSectRegStr = "([^\s]+).*"
    firstSect = re.match(firstSectRegStr, fenString).groups()[0]

    for ptype in PieceType:
        validChars = "012345678/%s%s" % (PIECELABELS[ptype][0], PIECELABELS[ptype][1])
        regstr = "[^%s]" % validChars
        thisSect = re.sub(regstr, "1", firstSect)

        thisBoard = BitBoard(ptype, fenString = thisSect)
        retval[ptype] = thisBoard

    return retval   



class BitBoard(object):
    """
    Class representing bitmap for a single piece type
    Filled with WHITE, CLEAR, or BLACK values

    IMPORTANT:
    indexing into, say, "b1" is done by using index (0,1), for the 0th row, and the "1st" (b) column
    """
    __slots__ = ["board", "pieceType"]


    ################################
    # CONSTRUCTOR
    ################################

    def __init__(self, pieceType, fenString = None):
        self.board = np.zeros((8,8)).astype(np.int8)
        self.pieceType = pieceType

        if fenString:
            self.constructFromFen(fenString)

    @classmethod
    def copy(cls, other):
        retval = BitBoard(other.pieceType)
        retval.board = np.copy(other.board)
        return retval
        
    ################################
    # QUICK ACCESSORS
    ################################

    def __getitem__(self, key):
        """
        Allows indexing into our board array more or less directly
        """
        if key == ():
            return self
        return Occupier(self.board[key])

    def __setitem__(self, key, value):
        self.board[key] = Occupier(value)

    def getWhitePositions(self):
        rowPos, colPos = np.where(np.equal(self.board, Occupier.WHITE))
        return list(zip(rowPos, colPos))

    def getBlackPositions(self):
        rowPos, colPos = np.where(np.equal(self.board, Occupier.BLACK))
        return list(zip(rowPos, colPos))

    def getAllPositions(self):
        rowPos, colPos = np.where(np.logical_not(np.equal(self.board, Occupier.CLEAR)))
        return list(zip(rowPos, colPos))

    def getPositionsOf(self, color):
        if color == Occupier.WHITE:
            return self.getWhitePositions()
        elif color == Occupier.BLACK:
            return self.getBlackPositions()
        elif color == Occupier.CLEAR:
            return self.getAllPositions()
        else:
            raise ValueError
 
    ################################
    # STRING REPRESENTATIONS
    ################################

    def constructFromFen(self, fenString):
        """
        Makes bitboard match the given fen string
        assumes any capital letter is white, lowercase is black
        """
        #TODO: WAAAAAAY more error checking
        regstr = FENParseString
        fenMatches = re.match(regstr, fenString).groups()

        for i in range(8):
            thisRowFen = fenMatches[7 - i]
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
        This gets called by __repr__, so if we're doing it a lot,
        we should be wary of the computational complexity, probably
        """
        retval = ""
        for i in range(self.board.shape[0] - 1, 0, -1):
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
            #if i != self.board.shape[0] - 1:
            if i != 0:
                retval += '/'

        return retval

    def __str__(self):
        """
        Note: Board may look flipped due to numpy array indexing
        """
        retval = "<BitBoard %s\n%s>" % (self.pieceType.name, self.board)
        return retval

    def __repr__(self):
        retval = "<BitBoard %s>" % self.asFen()
        return retval
    
