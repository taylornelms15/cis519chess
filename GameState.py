import logging
import enum
import re
import numpy as np

from log import setupLogging
from BitBoard import BitBoard, PieceType, BitBoardsFromFenString, FENParseString, S2I, I2S, Occupier, PIECELABELS
import MoveSquares

import pdb


class BadGameParseException(Exception):
    pass


class Turn(enum.IntEnum):
    WHITE = 1
    BLACK = -1


class Castle(enum.IntEnum):
    """ 
    Four different castle moves:
        White Kingside
        White Queenside
        Black Kingside
        Black Queenside
    Useful for indexing into arrays, probably
    """
    WKING = 0
    WQUEEN = 1
    BKING = 2
    BQUEEN = 3


class GameState(object):
    __slots__ = ['bitboards', 'turn',
                 'possibleCastles', 'halfmoveClock', 'enpassant']
    initialBoardFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def __init__(self, startingFrom=None, fenString=None):
        if startingFrom:
            # should do a copy of these, rather than by-reference
            self.bitboards = [BitBoard.copy(x) for x in startingFrom.bitboards]
            if startingFrom.turn == Turn.WHITE:
                self.turn = Turn.BLACK
            else:
                self.turn = Turn.WHITE
            self.possibleCastles = [x for x in startingFrom.possibleCastles]
            self.halfmoveClock = startingFrom.halfmoveClock
            self.enpassant = None
        elif fenString:  # If starting from a fenstring representation of a game state
            self.constructFromFenString(fenString)
        else:
            # bitfields representing piece positions
            self.bitboards = [BitBoard() for x in PieceType]
            self.turn = Turn.WHITE  # whose turn it is
            # whether any of the four castle types can be done (only cares about whether the relevant pieces have moved previously, not the other castling rules)
            self.possibleCastles = [True, True, True, True]
            # number of half-moves (turns) since last pawn move or piece caputre, for determining draw (50-turn rule)
            self.halfmoveClock = 0
            self.enpassant = None

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
        enpassantString = fenPieces[10]
        if (enpassantString != '-'):
            self.enpassant = S2I(enpassantString)
        else:
            self.enpassant = None
        self.possibleCastles = self.possCastlesFromCastleFen(castleString)
        self.halfmoveClock = 0  # TODO: actually read this

    def possCastlesFromCastleFen(self, castleString):
        retval = [False, False, False, False]
        if "K" in castleString:
            retval[Castle.WKING] = True
        if "Q" in castleString:
            retval[Castle.WQUEEN] = True
        if "k" in castleString:
            retval[Castle.BKING] = True
        if "q" in castleString:
            retval[Castle.BQUEEN] = True
        return retval

    @classmethod
    def getInitialState(cls):
        """
        Class constructor that creates the state of the initial chess board
        """
        return GameState(fenString=cls.initialBoardFen)

    # Functions to treat a GameState more atomically in dictionaries and whatnot

    def __repr__(self):
        retval = "<GameState %s>" % list(self._key())
        return retval

    def _key(self):
        # , self.halfmoveClock)
        return (tuple(self.bitboards), self.turn, tuple(self.possibleCastles))

    def __hash__(self):
        return hash(self._key())

    def __eq__(self, other):
        if isinstance(other, GameState):
            return self._key() == other._key()
        return NotImplemented

    # Some handy representational tools
    def prettyDebug(self):
        for i in range(7, -1, -1):
            line = str(i + 1) + "| "
            for j in range(0, 8):
                color, pieceType = self.getPieceAtPosition((i, j))
                if color == Occupier.CLEAR:
                    token = "-"
                else:
                    token = PIECELABELS[pieceType][0 if color ==
                                                   Occupier.WHITE else 1]
                line += token
                line += " "
            print(line)
        print("   a b c d e f g h")

    # Accessors

    def getBitboards(self):
        return self.bitboards

    def getTurn(self):
        return self.turn

    def getPossibleCastles(self):
        return self.possibleCastles

    def getHalfmoveClock(self):
        return self.halfmoveClock

    def getPieceAtPosition(self, pos):
        """
        @return tuple of (color, pieceType)
        """
        if isinstance(pos, str):
            pos = S2I(pos)
        elif isinstance(pos, np.ndarray):
            pos = tuple(pos)

        for i, board in enumerate(self.getBitboards()):
            if isinstance(board, np.ndarray):
                board = board[()]
            val = board[pos]
            if val != Occupier.CLEAR:
                return board[pos], PieceType(i)
        return Occupier.CLEAR, None

    def getPawns(self, color=Occupier.CLEAR):
        return self.getPiecesOfColor(self, PieceType.PAWN, color)

    def getRooks(self, color=Occupier.CLEAR):
        return self.getPiecesOfColor(self, PieceType.ROOK, color)

    def getKnights(self, color=Occupier.CLEAR):
        return self.getPiecesOfColor(self, PieceType.KNIGHT, color)

    def getBishops(self, color=Occupier.CLEAR):
        return self.getPiecesOfColor(self, PieceType.BISHOP, color)

    def getQueens(self, color=Occupier.CLEAR):
        return self.getPiecesOfColor(self, PieceType.QUEEN, color)

    def getKings(self, color=Occupier.CLEAR):
        return self.getPiecesOfColor(self, PieceType.KING, color)

    def getPiecesOfColor(self, pieceType, color=Occupier.CLEAR):
        board = self.bitboards[pieceType]
        if isinstance(board, np.ndarray):
            board = board[()]
        return board.getPositionsOf(color)

    # Mutators

    def incHalfmoveClock(self):
        self.halfmoveClock += 1

    def toggleTurn(self):
        if self.turn == Turn.WHITE:
            self.turn = Turn.BLACK
        else:
            self.turn = Turn.WHITE

    def modCastleBecauseMove(self, pos):
        """
        Modifies the castle table because the piece in the given position moved
        """
        if isinstance(pos, str):
            pos = S2I(pos)
        if pos == S2I("a1"):
            self.possibleCastles[Castle.WQUEEN] = False
        elif pos == S2I("e1"):
            self.possibleCastles[Castle.WKING] = False
            self.possibleCastles[Castle.WQUEEN] = False
        elif pos == S2I("h1"):
            self.possibleCastles[Castle.WKING] = False
        elif pos == S2I("a8"):
            self.possibleCastles[Castle.BQUEEN] = False
        elif pos == S2I("e8"):
            self.possibleCastles[Castle.BKING] = False
            self.possibleCastles[Castle.BQUEEN] = False
        elif pos == S2I("h8"):
            self.possibleCastles[Castle.BKING] = False

    def clearPieceInSpace(self, pos):
        if isinstance(pos, str):
            pos = S2I(pos)

        boards = self.getBitboards()
        for board in boards:
            if isinstance(board, np.ndarray):
                board = board[()]
            board[pos] = Occupier.CLEAR

    def putPieceInSpace(self, piece, color, pos):
        if isinstance(pos, str):
            pos = S2I(pos)
        if isinstance(piece, str):
            logging.error(piece)
        board = self.getBitboards()[piece]
        if isinstance(board, np.ndarray):
            board = board[()]
        board[pos] = color

    # Chess-specific functions

    def getPossibleMoves(self, board):
        """
        Given the current game state, returns a list of all possible moves that could be performed
        """
        # NOTE: can use the MoveSquares functions to help with this

        legal_moves = []

        for move in board.legal_moves:
            move_str = move.uci()
            start_loc = move_str[:2]
            end_loc = move_str[2:]

            my_move = Move(start_loc, end_loc)
            legal_moves.append(my_move)

        return legal_moves

        # raise NotImplementedError

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

    def __init__(self, startLoc, endLoc, castle=None, promotion=None):
        if isinstance(startLoc, str):
            startLoc = S2I(startLoc)
        if isinstance(endLoc, str):
            endLoc = S2I(endLoc)
        self.startLoc = startLoc
        self.endLoc = endLoc
        self.castle = castle
        self.promotion = promotion

    # FOR ABUSING DICTIONARIES

    def __eq__(self, other):
        attrs = self.__slots__
        for attr in attrs:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def _key(self):
        return (self.startLoc, self.endLoc, self.castle, self.promotion)

    def __hash__(self):
        return hash(self._key())

    # STRING REPRESENTATIONS
    def __repr__(self):
        if self.castle != None:
            retval = "<Move C:%s>" % self.castle.name
        else:
            retval = "<Move %s->%s>" % (I2S(self.startLoc), I2S(self.endLoc))
        return retval

    # ACCESSORS

    def startsAt(self, startLoc):
        if isinstance(startLoc, int):
            startLoc = (startLoc / 8, startLoc % 8)
        elif isinstance(startLoc, str):
            startLoc = S2I(startLoc)
        return startLoc == self.startLoc

    def endsAt(self, endLoc):
        if isinstance(endLoc, int):
            endLoc = (endLoc / 8, endLoc % 8)
        elif isinstance(endLoc, str):
            endLoc = S2I(endLoc)
        return endLoc == self.endLoc

    def isCastle(self):
        retval = self.castle != None
        return retval

    # SPECIAL CONSTRUCTORS

    @classmethod
    def constructCastle(cls, castleType):
        return Move(None, None, castleType, None)

    @classmethod
    def constructFromPgnHalfmove(cls, gameState, Piece, Rank, File, Endloc, Promotion):
        # remember: order is (file)(rank) (files are letters, ranks are numbers)
        endLoc = S2I(Endloc)
        if Piece == '':
            Piece = PieceType.PAWN
        else:
            matchingTypes = [
                x for x in PIECELABELS.keys() if PIECELABELS[x][0] == Piece]
            Piece = matchingTypes[0]
        if Promotion != '':
            matchingTypes = [x for x in PIECELABELS.keys(
            ) if PIECELABELS[x][0] == Promotion]
            Promotion = matchingTypes[0]
        # figure out what fucking piece is moving
        possibleMovers = gameState.getPiecesOfColor(Piece, gameState.turn)
        candidates = []
        for candidate in possibleMovers:
            mask = MoveSquares.makeMoveMask(
                Piece, gameState.turn, candidate, gameState)
            if mask[endLoc]:
                candidates.append(candidate)

        if len(candidates) > 1:  # if piece ambiguity
            if File != '':
                candidates = [x for x in candidates if I2S(x)[0] == File]
            if Rank != '':
                candidates = [x for x in candidates if I2S(x)[1] == Rank]

        if len(candidates) != 1:
            # pdb.set_trace()
            #raise ValueError("Still have piece ambiguity")
            raise BadGameParseException(
                "Some sort of piece ambiguity or emptiness")

        startLoc = candidates[0]

        retval = Move(startLoc, endLoc, None, Promotion)
        return retval

    # APPLYING GAME STATE TRANSFORMATIONS

    def _applyCastle(self, gameState):
        retval = GameState(startingFrom=gameState)
        retval.incHalfmoveClock()
        if self.castle == Castle.WKING:
            retval.clearPieceInSpace("e1")
            retval.clearPieceInSpace("h1")
            retval.putPieceInSpace(PieceType.KING, Occupier.WHITE, "g1")
            retval.putPieceInSpace(PieceType.ROOK, Occupier.WHITE, "f1")
            retval.modCastleBecauseMove("e1")
            retval.modCastleBecauseMove("h1")
            return retval
        elif self.castle == Castle.WQUEEN:
            retval.clearPieceInSpace("e1")
            retval.clearPieceInSpace("a1")
            retval.putPieceInSpace(PieceType.KING, Occupier.WHITE, "c1")
            retval.putPieceInSpace(PieceType.ROOK, Occupier.WHITE, "d1")
            retval.modCastleBecauseMove("e1")
            retval.modCastleBecauseMove("a1")
            return retval
        elif self.castle == Castle.BKING:
            retval.clearPieceInSpace("e8")
            retval.clearPieceInSpace("h8")
            retval.putPieceInSpace(PieceType.KING, Occupier.BLACK, "g8")
            retval.putPieceInSpace(PieceType.ROOK, Occupier.BLACK, "f8")
            retval.modCastleBecauseMove("e8")
            retval.modCastleBecauseMove("h8")
            return retval
        elif self.castle == Castle.BQUEEN:
            retval.clearPieceInSpace("e8")
            retval.clearPieceInSpace("a8")
            retval.putPieceInSpace(PieceType.KING, Occupier.BLACK, "c8")
            retval.putPieceInSpace(PieceType.ROOK, Occupier.BLACK, "d8")
            retval.modCastleBecauseMove("e8")
            retval.modCastleBecauseMove("a8")
            return retval
        else:
            raise ValueError(
                "Applying a castle move in not a castle move (????)")

    def apply(self, gameState):
        """
        Applies the move of the piece from startLoc to endLoc
        Does not check if the move is legal

        @return The GameState after this move is applied
        """
        if self.castle != None:
            return self._applyCastle(gameState)

        retval = GameState(startingFrom=gameState)
        retval.incHalfmoveClock()

        # First: find what's moving (pick up piece)
        color, pieceType = retval.getPieceAtPosition(self.startLoc)
        if color == Occupier.CLEAR:
            raise ValueError("Didn't find a piece at the startloc to move?")

        # Clear spots for src and dest
        retval.clearPieceInSpace(self.startLoc)
        retval.modCastleBecauseMove(self.startLoc)
        retval.clearPieceInSpace(self.endLoc)

        # Put piece in correct spot
        if self.promotion != None and self.promotion != '':
            pieceType = self.promotion
        retval.putPieceInSpace(pieceType, color, self.endLoc)

        return retval



def main():
    myState = GameState.getInitialState()
    logging.info(myState)

    print(S2I("e2"))


if __name__ == "__main__":
    setupLogging()
    main()
