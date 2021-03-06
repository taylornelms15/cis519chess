"""
@file   MoveSquares.py
@author Taylor Nelms
Encoding of where different pieces can go from a particular position
Steals tables from LeelaChessZero
"""

import logging
import sys
import numpy as np

from log import setupLogging
from BitBoard import Occupier, PieceType

import pdb

ROOK_DIRS = np.array([ [0, 1], [1, 0], [0, -1], [-1, 0] ], dtype=np.int8)
BISHOP_DIRS = np.array([ [1, 1], [1, -1], [-1, 1], [-1, -1] ], dtype=np.int8)
QUEEN_DIRS = np.concatenate((ROOK_DIRS, BISHOP_DIRS), axis=0)
KNIGHT_DIRS = np.array([ [1, 2], [2, 1], [1, -2], [2, -1], [-1, 2], [-2, 1], [-1, -2], [-2, -1] ], dtype = np.int8)
PAWN_WM_DIRS = np.array([ [1, 0] ], dtype=np.int8)
PAWN_BM_DIRS = np.array([ [-1, 0] ], dtype=np.int8)
PAWN_WC_DIRS = np.array([ [1, 1], [1, -1] ], dtype=np.int8)
PAWN_BC_DIRS = np.array([ [-1, -1], [-1, 1] ], dtype=np.int8)

def indexInBounds(idx, direction = np.array([0, 0])):
    return np.all(np.logical_and(np.greater_equal(idx + direction, 0), np.less(idx + direction, 8)))

def _makeMoveMaskPawnB(idx, gameState):
    retval = np.zeros((8, 8), dtype=np.bool)

    #Move directions
    if indexInBounds(idx, PAWN_BM_DIRS[0]):
        idxTemp = idx + PAWN_BM_DIRS[0]
        occupierColor, _ = gameState.getPieceAtPosition(idxTemp)
        if occupierColor == Occupier.CLEAR:
            retval[tuple(idxTemp)] = True
            if idx[0] == 6:
                idxTemp = idxTemp + PAWN_BM_DIRS[0]
                occupierColor, _ = gameState.getPieceAtPosition(idxTemp)
                if occupierColor == Occupier.CLEAR:
                    retval[tuple(idxTemp)] = True
    #capture directions
    for direction in PAWN_BC_DIRS:
        if indexInBounds(idx, direction):
            idxTemp = idx + direction
            occupierColor, _ = gameState.getPieceAtPosition(idxTemp)
            if occupierColor == Occupier.WHITE:
                retval[tuple(idxTemp)] = True
    return retval

def _makeMoveMaskPawnW(idx, gameState):
    retval = np.zeros((8, 8), dtype=np.bool)

    #Move directions
    if indexInBounds(idx, PAWN_WM_DIRS[0]):
        idxTemp = idx + PAWN_WM_DIRS[0]
        occupierColor, _ = gameState.getPieceAtPosition(idxTemp)
        if occupierColor == Occupier.CLEAR:
            retval[tuple(idxTemp)] = True
            if idx[0] == 1:
                idxTemp = idxTemp + PAWN_WM_DIRS[0]
                occupierColor, _ = gameState.getPieceAtPosition(idxTemp)
                if occupierColor == Occupier.CLEAR:
                    retval[tuple(idxTemp)] = True
    #capture directions
    for direction in PAWN_WC_DIRS:
        if indexInBounds(idx, direction):
            idxTemp = idx + direction
            occupierColor, _ = gameState.getPieceAtPosition(idxTemp)
            if occupierColor == Occupier.BLACK:
                retval[tuple(idxTemp)] = True
    return retval

def _makeMoveMaskPawn(color, idx, gameState):
    if color == Occupier.WHITE:
        return _makeMoveMaskPawnW(idx, gameState)
    elif color == Occupier.BLACK:
        return _makeMoveMaskPawnB(idx, gameState)
    else:
        raise ValueError

def _makeMoveMaskKing(color, idx, gameState):
    retval = np.zeros((8, 8), dtype=np.bool)
    for direction in QUEEN_DIRS:
        idxTemp = np.copy(idx)
        if indexInBounds(idxTemp, direction):
            idxTemp += direction
            occupierColor, _ = gameState.getPieceAtPosition(idxTemp)
            if occupierColor == Occupier.CLEAR:
                retval[tuple(idxTemp)] = True
            elif occupierColor == Occupier.WHITE:
                if color == Occupier.BLACK:
                    retval[tuple(idxTemp)] = True
            elif occupierColor == Occupier.BLACK:
                if color == Occupier.WHITE:
                    retval[tuple(idxTemp)] = True
            else:
                raise ValueError
    return retval

def _makeMoveMaskKnight(color, idx, gameState):
    retval = np.zeros((8, 8), dtype=np.bool)
    for direction in KNIGHT_DIRS:
        idxTemp = np.copy(idx)
        if indexInBounds(idxTemp, direction):
            idxTemp += direction
            occupierColor, _ = gameState.getPieceAtPosition(idxTemp)
            if occupierColor == Occupier.CLEAR:
                retval[tuple(idxTemp)] = True
            elif occupierColor == Occupier.WHITE:
                if color == Occupier.BLACK:
                    retval[tuple(idxTemp)] = True
            elif occupierColor == Occupier.BLACK:
                if color == Occupier.WHITE:
                    retval[tuple(idxTemp)] = True
            else:
                raise ValueError
    return retval

def _makeMoveMaskRook(color, idx, gameState):
    retval = np.zeros((8, 8), dtype=np.bool)
    for direction in ROOK_DIRS:
        idxTemp = np.copy(idx)
        while indexInBounds(idxTemp, direction):
            idxTemp += direction
            occupierColor, _ = gameState.getPieceAtPosition(idxTemp)
            if occupierColor == Occupier.CLEAR:
                retval[tuple(idxTemp)] = True
            elif occupierColor == Occupier.WHITE:
                if color == Occupier.BLACK:
                    retval[tuple(idxTemp)] = True
                break
            elif occupierColor == Occupier.BLACK:
                if color == Occupier.WHITE:
                    retval[tuple(idxTemp)] = True
                break
            else:
                raise ValueError
    return retval

def _makeMoveMaskBishop(color, idx, gameState):
    retval = np.zeros((8, 8), dtype=np.bool)
    for direction in BISHOP_DIRS:
        idxTemp = np.copy(idx)
        while indexInBounds(idxTemp, direction):
            idxTemp += direction
            occupierColor, _ = gameState.getPieceAtPosition(idxTemp)
            if occupierColor == Occupier.CLEAR:
                retval[tuple(idxTemp)] = True
            elif occupierColor == Occupier.WHITE:
                if color == Occupier.BLACK:
                    retval[tuple(idxTemp)] = True
                break
            elif occupierColor == Occupier.BLACK:
                if color == Occupier.WHITE:
                    retval[tuple(idxTemp)] = True
                break
            else:
                raise ValueError
    return retval

def _makeMoveMaskQueen(color, idx, gameState):
    retval = np.zeros((8, 8), dtype=np.bool)
    for direction in QUEEN_DIRS:
        idxTemp = np.copy(idx)
        while indexInBounds(idxTemp, direction):
            idxTemp += direction
            occupierColor, _ = gameState.getPieceAtPosition(idxTemp)
            if occupierColor == Occupier.CLEAR:
                retval[tuple(idxTemp)] = True
            elif occupierColor == Occupier.WHITE:
                if color == Occupier.BLACK:
                    retval[tuple(idxTemp)] = True
                break
            elif occupierColor == Occupier.BLACK:
                if color == Occupier.WHITE:
                    retval[tuple(idxTemp)] = True
                break
            else:
                raise ValueError
    return retval

def makeMoveMask(pieceType, color, idx, gameState):
    """
    Given a piece type at a given index, creates a binary mask of where that piece could move
    """
    if (pieceType == PieceType.PAWN):
        return _makeMoveMaskPawn(color, idx, gameState)
    elif (pieceType == PieceType.ROOK):
        return _makeMoveMaskRook(color, idx, gameState)
    elif (pieceType == PieceType.KNIGHT):
        return _makeMoveMaskKnight(color, idx, gameState)
    elif (pieceType == PieceType.BISHOP):
        return _makeMoveMaskBishop(color, idx, gameState)
    elif (pieceType == PieceType.QUEEN):
        return _makeMoveMaskQueen(color, idx, gameState)
    elif (pieceType == PieceType.KING):
        return _makeMoveMaskKing(color, idx, gameState)


def main():
    #state = GameState.GameState(fenString = exampleWhiteCheckmate)
    state = GameState.GameState(fenString = GameState.GameState.initialBoardFen)
    logging.info(state)
    print("Pawn mask: %s" % makeMoveMask(PieceType.PAWN, Occupier.WHITE, (3, 4), state))
    print("Rook mask: %s" % makeMoveMask(PieceType.ROOK, Occupier.WHITE, (3, 4), state))
    print("Knight mask: %s" % makeMoveMask(PieceType.KNIGHT, Occupier.WHITE, (3, 4), state))
    print("Bishop mask: %s" % makeMoveMask(PieceType.BISHOP, Occupier.WHITE, (3, 4), state))
    print("Queen mask: %s" % makeMoveMask(PieceType.QUEEN, Occupier.WHITE, (3, 4), state))
    print("King mask: %s" % makeMoveMask(PieceType.KING, Occupier.WHITE, (3, 4), state))
     

if __name__ == "__main__":
    setupLogging()
    import GameState
    from mcts import exampleWhiteCheckmate
    main()


