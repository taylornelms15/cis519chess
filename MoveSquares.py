"""
@file   MoveSquares.py
@author Taylor Nelms
Encoding of where different pieces can go from a particular position
Steals tables from LeelaChessZero
"""

import logging
import sys
import numpy as np

from BitBoard import Occupier, PieceType

import pdb

ROOK_DIRS = np.array([ [0, 1], [1, 0], [0, -1], [-1, 0] ], dtype=np.int8)
BISHOP_DIRS = np.array([ [1, 1], [1, -1], [-1, 1], [-1, -1] ], dtype=np.int8)
QUEEN_DIRS = np.concatenate((ROOK_DIRS, BISHOP_DIRS), axis=0)
KNIGHT_DIRS = np.array([ [1, 2], [2, 1], [1, -2], [2, -1], [-1, 2], [-2, 1], [-1, -2], [-2, -1], dtype = np.int8)
PAWN_WM_DIRS = np.array([ [0, 1] ], dtype=np.int8)
PAWN_BM_DIRS = np.array([ [0, -1] ], dtype=np.int8)
PAWN_WC_DIRS = np.array([ [-1, 1], [1, 1] ], dtype=np.int8)
PAWN_BC_DIRS = np.array([ [-1, -1], [1, -1] ], dtype=np.int8)

def indexInBounds(idx, direction = np.array([0, 0])):
    return np.all(np.logical_and(np.greater_equal(0, idx + direction), np.less(8, idx + direction))):

def _makeMoveMaskPawnB(idx, gameState):
    retval = np.zeros((8, 8), dtype=np.bool)

    #Move directions
    if indexInBounds(idx, PAWN_BM_DIRS[0]):
        idxTemp = idx + PAWN_BM_DIRS[0]
        occupierColor, _ = gameState.getPieceAtPosition(idxTemp)
        if occupierColor == Occupier.CLEAR:
            retval[idxTemp] = True
            if idx[1] == 6:
                idxTemp = idxTemp + PAWN_BM_DIRS[0]
                occupierColor, _ = gameState.getPieceAtPosition(idxTemp)
                if occupierColor == Occupier.CLEAR:
                    retval[idxTemp] = True
    #capture directions
    for direction in PAWN_BC_DIRS:
        if indexInBounds(idx, direction):
            idxTemp = idx + direction
            occupierColor, _ = gameState.getPieceAtPosition(idxTemp)
            if occupierColor == Occupier.WHITE:
                retval[idxTemp] = True
    return retval

def _makeMoveMaskPawnW(idx, gameState):
    retval = np.zeros((8, 8), dtype=np.bool)

    #Move directions
    if indexInBounds(idx, PAWN_WM_DIRS[0]):
        idxTemp = idx + PAWN_WM_DIRS[0]
        occupierColor, _ = gameState.getPieceAtPosition(idxTemp)
        if occupierColor == Occupier.CLEAR:
            retval[idxTemp] = True
            if idx[1] == 1:
                idxTemp = idxTemp + PAWN_WM_DIRS[0]
                occupierColor, _ = gameState.getPieceAtPosition(idxTemp)
                if occupierColor == Occupier.CLEAR:
                    retval[idxTemp] = True
    #capture directions
    for direction in PAWN_WC_DIRS:
        if indexInBounds(idx, direction):
            idxTemp = idx + direction
            occupierColor, _ = gameState.getPieceAtPosition(idxTemp)
            if occupierColor == Occupier.BLACK:
                retval[idxTemp] = True
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
                retval[idxTemp] = True
            elif occupierColor == Occupier.WHITE:
                if color == Occupier.BLACK:
                    retval[idxTemp] = True
                break
            elif occupierColor == Occupier.BLACK:
                if color == Occupier.WHITE:
                    retval[idxTemp] = True
                break
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
                retval[idxTemp] = True
            elif occupierColor == Occupier.WHITE:
                if color == Occupier.BLACK:
                    retval[idxTemp] = True
                break
            elif occupierColor == Occupier.BLACK:
                if color == Occupier.WHITE:
                    retval[idxTemp] = True
                break
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
                retval[idxTemp] = True
            elif occupierColor == Occupier.WHITE:
                if color == Occupier.BLACK:
                    retval[idxTemp] = True
                break
            elif occupierColor == Occupier.BLACK:
                if color == Occupier.WHITE:
                    retval[idxTemp] = True
                break
            else:
                raise ValueError
    return retval

def _makeMoveMaskBishop(color, idx, gameState):
    retval = np.zeros((8, 8), dtype=np.bool)
    for direction in BISHOP_DIRS:
        idxTemp = np.copy(idx)
        while np.all(np.logical_and(np.greater_equal(0, idxTemp + direction), np.less(8, idxTemp + direction))):
            idxTemp += direction
            occupierColor, _ = gameState.getPieceAtPosition(idxTemp)
            if occupierColor == Occupier.CLEAR:
                retval[idxTemp] = True
            elif occupierColor == Occupier.WHITE:
                if color == Occupier.BLACK:
                    retval[idxTemp] = True
                break
            elif occupierColor == Occupier.BLACK:
                if color == Occupier.WHITE:
                    retval[idxTemp] = True
                break
            else:
                raise ValueError
    return retval

def _makeMoveMaskQueen(color, idx, gameState):
    retval = np.zeros((8, 8), dtype=np.bool)
    for direction in QUEEN_DIRS:
        idxTemp = np.copy(idx)
        while np.all(np.logical_and(np.greater_equal(0, idxTemp + direction), np.less(8, idxTemp + direction))):
            idxTemp += direction
            occupierColor, _ = gameState.getPieceAtPosition(idxTemp)
            if occupierColor == Occupier.CLEAR:
                retval[idxTemp] = True
            elif occupierColor == Occupier.WHITE:
                if color == Occupier.BLACK:
                    retval[idxTemp] = True
                break
            elif occupierColor == Occupier.BLACK:
                if color == Occupier.WHITE:
                    retval[idxTemp] = True
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





"""
def maskFromTable(table, idx):
    val = int(table[idx])
    valB = bytearray(val.to_bytes(8, sys.byteorder))
    valNumpy = np.unpackbits(valB).reshape((8, 8))
    valNumpy = np.fliplr(valNumpy)
    return valNumpy.astype(np.bool)


RookAttacks = np.array([   [0x01010101010101FE, 0x02020202020202FD, 0x04040404040404FB, 0x08080808080808F7, 
                            0x10101010101010EF, 0x20202020202020DF, 0x40404040404040BF, 0x808080808080807F], 
                           [0x010101010101FE01, 0x020202020202FD02, 0x040404040404FB04, 0x080808080808F708,
                            0x101010101010EF10, 0x202020202020DF20, 0x404040404040BF40, 0x8080808080807F80], 
                           [0x0101010101FE0101, 0x0202020202FD0202, 0x0404040404FB0404, 0x0808080808F70808, 
                            0x1010101010EF1010, 0x2020202020DF2020, 0x4040404040BF4040, 0x80808080807F8080],
                           [0x01010101FE010101, 0x02020202FD020202, 0x04040404FB040404, 0x08080808F7080808, 
                            0x10101010EF101010, 0x20202020DF202020, 0x40404040BF404040, 0x808080807F808080],
                           [0x010101FE01010101, 0x020202FD02020202, 0x040404FB04040404, 0x080808F708080808,
                            0x101010EF10101010, 0x202020DF20202020, 0x404040BF40404040, 0x8080807F80808080], 
                           [0x0101FE0101010101, 0x0202FD0202020202, 0x0404FB0404040404, 0x0808F70808080808, 
                            0x1010EF1010101010, 0x2020DF2020202020, 0x4040BF4040404040, 0x80807F8080808080],
                           [0x01FE010101010101, 0x02FD020202020202, 0x04FB040404040404, 0x08F7080808080808,
                            0x10EF101010101010, 0x20DF202020202020, 0x40BF404040404040, 0x807F808080808080], 
                           [0xFE01010101010101, 0xFD02020202020202, 0xFB04040404040404, 0xF708080808080808,
                            0xEF10101010101010, 0xDF20202020202020, 0xBF40404040404040, 0x7F80808080808080] ], dtype=np.uint64 )
RookMoves = RookAttacks

BishopAttacks = np.array([ [0x8040201008040200, 0x0080402010080500, 0x0000804020110A00, 0x0000008041221400, 
                            0x0000000182442800, 0x0000010204885000, 0x000102040810A000, 0x0102040810204000], 
                           [0x4020100804020002, 0x8040201008050005, 0x00804020110A000A, 0x0000804122140014,
                            0x0000018244280028, 0x0001020488500050, 0x0102040810A000A0, 0x0204081020400040], 
                           [0x2010080402000204, 0x4020100805000508, 0x804020110A000A11, 0x0080412214001422, 
                            0x0001824428002844, 0x0102048850005088, 0x02040810A000A010, 0x0408102040004020],
                           [0x1008040200020408, 0x2010080500050810, 0x4020110A000A1120, 0x8041221400142241, 
                            0x0182442800284482, 0x0204885000508804, 0x040810A000A01008, 0x0810204000402010], 
                           [0x0804020002040810, 0x1008050005081020, 0x20110A000A112040, 0x4122140014224180,
                            0x8244280028448201, 0x0488500050880402, 0x0810A000A0100804, 0x1020400040201008], 
                           [0x0402000204081020, 0x0805000508102040, 0x110A000A11204080, 0x2214001422418000, 
                            0x4428002844820100, 0x8850005088040201, 0x10A000A010080402, 0x2040004020100804],
                           [0x0200020408102040, 0x0500050810204080, 0x0A000A1120408000, 0x1400142241800000, 
                            0x2800284482010000, 0x5000508804020100, 0xA000A01008040201, 0x4000402010080402], 
                           [0x0002040810204080, 0x0005081020408000, 0x000A112040800000, 0x0014224180000000,
                            0x0028448201000000, 0x0050880402010000, 0x00A0100804020100, 0x0040201008040201] ], dtype=np.uint64)
BishopMoves = BishopAttacks

KnightAttacks = np.array([ [0x0000000000020400, 0x0000000000050800, 0x00000000000A1100, 0x0000000000142200, 
                            0x0000000000284400, 0x0000000000508800, 0x0000000000A01000, 0x0000000000402000], 
                           [0x0000000002040004, 0x0000000005080008, 0x000000000A110011, 0x0000000014220022,
                            0x0000000028440044, 0x0000000050880088, 0x00000000A0100010, 0x0000000040200020], 
                           [0x0000000204000402, 0x0000000508000805, 0x0000000A1100110A, 0x0000001422002214, 
                            0x0000002844004428, 0x0000005088008850, 0x000000A0100010A0, 0x0000004020002040],
                           [0x0000020400040200, 0x0000050800080500, 0x00000A1100110A00, 0x0000142200221400, 
                            0x0000284400442800, 0x0000508800885000, 0x0000A0100010A000, 0x0000402000204000], 
                           [0x0002040004020000, 0x0005080008050000, 0x000A1100110A0000, 0x0014220022140000,
                            0x0028440044280000, 0x0050880088500000, 0x00A0100010A00000, 0x0040200020400000], 
                           [0x0204000402000000, 0x0508000805000000, 0x0A1100110A000000, 0x1422002214000000, 
                            0x2844004428000000, 0x5088008850000000, 0xA0100010A0000000, 0x4020002040000000],
                           [0x0400040200000000, 0x0800080500000000, 0x1100110A00000000, 0x2200221400000000, 
                            0x4400442800000000, 0x8800885000000000, 0x100010A000000000, 0x2000204000000000], 
                           [0x0004020000000000, 0x0008050000000000, 0x00110A0000000000, 0x0022140000000000,
                            0x0044280000000000, 0x0088500000000000, 0x0010A00000000000, 0x0020400000000000] ], dtype=np.uint64)
KnightMoves = KnightAttacks

QueenAttacks = np.logical_or(RookAttacks, BishopAttacks)
QueenMoves = QueenAttacks

PawnAttacks = np.array([   [0x0000000000000200, 0x0000000000000500, 0x0000000000000A00, 0x0000000000001400, 
                            0x0000000000002800, 0x0000000000005000, 0x000000000000A000, 0x0000000000004000], 
                           [0x0000000000020000, 0x0000000000050000, 0x00000000000A0000, 0x0000000000140000,
                            0x0000000000280000, 0x0000000000500000, 0x0000000000A00000, 0x0000000000400000], 
                           [0x0000000002000000, 0x0000000005000000, 0x000000000A000000, 0x0000000014000000, 
                            0x0000000028000000, 0x0000000050000000, 0x00000000A0000000, 0x0000000040000000],
                           [0x0000000200000000, 0x0000000500000000, 0x0000000A00000000, 0x0000001400000000, 
                            0x0000002800000000, 0x0000005000000000, 0x000000A000000000, 0x0000004000000000], 
                           [0x0000020000000000, 0x0000050000000000, 0x00000A0000000000, 0x0000140000000000,
                            0x0000280000000000, 0x0000500000000000, 0x0000A00000000000, 0x0000400000000000], 
                           [0x0002000000000000, 0x0005000000000000, 0x000A000000000000, 0x0014000000000000, 
                            0x0028000000000000, 0x0050000000000000, 0x00A0000000000000, 0x0040000000000000],
                           [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 
                            0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000], 
                           [0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
                            0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000] ], dtype = np.uint64)
PawnMoves = PawnAttacks
"""







