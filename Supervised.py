"""
@file   Supervised.py
@author Taylor Nelms

Handles all the torch-related parts of our networks
"""

import logging
import enum
import re
import numpy as np
import torch

from BitBoard import BitBoard, PieceType, S2I, I2S, PIECELABELS
from GameState import Turn, Castle, GameState, Move

import pdb

KSIDE = 64
QSIDE = 65

def moveToTensor(move):
    """
    Converts the Move object to a (set of?) tensor objects
    First attempt will turn it into two vectors: "from" and "to" positions
    first 64 elements are just the regular indexes
    Next two are castles (kingside, queenside)
    Not supporting distinguishing promotions (yet?)
    """
    if move.castle != None:
        if move.castle == Castle.WKING or move.castle == Castle.BKING:
            startLoc = KSIDE
            endLoc   = KSIDE
        else:
            startLoc = QSIDE
            endLoc   = QSIDE
    else:
        startLoc    = move.startLoc[0]  * 8 + move.startLoc[1]
        endLoc      = move.endLoc[0]    * 8 + move.endLoc[1]

    retStart            = np.zeros((66,), dtype = np.bool)
    retEnd              = np.zeros((66,), dtype = np.bool)
    retStart[startLoc]  = 1
    retEnd[endLoc]      = 1

    retStart    = torch.from_numpy(retStart)
    retEnd      = torch.from_numpy(retEnd)

    return [retStart, retEnd]

def gameStateToTensor(gameState):
    """
    Converts the GameState object to a (set of?) tensor objects

    Initial attempt: each bitboard is a "channel"
    Make one additional channel for the turn and castles metadata
    """
    bitBoards   = gameState.getBitboards()#[np.darray(64,64), ...](len 6) (type: actually a BitBoard)
    turn        = gameState.getTurn()#Turn (int)
    castles     = gameState.getPossibleCastles()#[bool, bool, bool, bool]
    halfmove    = gameState.getHalfmoveClock()#int (for now: unused)

    #notably: each should be within an int8 (halfmove would change this, but ignoring for now)

    #retval = np.array([turn, castles[0], castles[1], castles[2], castles[3], 0, 0, 0], dtype=np.int8).reshape((1, 8))
    firstLayer = np.zeros((8, 8), dtype=np.int8)
    #put the castles in the corner, turn in the middle (making this up)
    firstLayer[0:2, 0:2] = castles[Castle.WQUEEN]
    firstLayer[0:2, 6:8] = castles[Castle.WKING]
    firstLayer[6:8, 0:2] = castles[Castle.BQUEEN]
    firstLayer[6:8, 6:8] = castles[Castle.BKING]
    firstLayer[2:6, 2:6] = turn

    retval = np.stack([firstLayer] + [x.getRawBoard() for x in bitBoards], axis=0)

    return torch.from_numpy(retval)








def main():
    myState = GameState.getInitialState()
    myMove  = Move("b2", "b4")

    moveTensor  = moveToTensor(myMove)
    stateTensor = gameStateToTensor(myState)

    pdb.set_trace()


if __name__ == "__main__":
    from log import setupLogging
    setupLogging()
    main()

