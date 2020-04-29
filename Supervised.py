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
import argparse

import PGNIngest
from BitBoard import BitBoard, PieceType, S2I, I2S, PIECELABELS
from GameState import Turn, Castle, GameState, Move
from ChessNet import ChessNet, trainModel

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
    Update: will just make one tensor for output
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

    retval = np.concatenate((retStart, retEnd), axis=0)

    #retStart    = torch.from_numpy(retStart)
    #retEnd      = torch.from_numpy(retEnd)

    return torch.from_numpy(retval.astype(np.float))

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







def getTensorData(args):
    """
    Takes in parsed command line args, creates a set of tensor data for the (GameState, Move) pairs
    Returns a TensorDataset
    """
    if args.tensorData == None:

        rawData = []

        if (args.pgnFiles):
            for pgnFile in args.pgnFiles:
                rawData += PGNIngest.parseAllLinesInFile(pgnFile)
        elif (args.pickedFile):
            pass
            #this is where we would unpickle things into rawData

        """
        Currently, `rawData` is a 2d list
        Each entry is a list representing a single chess game that ended in a checkmate
        Each such game is a list of (GameState, move) pairs
        At some point, we may re-evaluate how we represent that set of raw data
        """
        states = []
        moves  = []

        for gameListing in rawData:
            otherWay = list(zip(*gameListing))
            states += list(otherWay[0])
            moves  += list(otherWay[1])

        states = torch.stack([gameStateToTensor(x) for x in states])
        moves  = torch.stack([moveToTensor(x)      for x in moves])

        fullDataset = torch.utils.data.TensorDataset(states, moves)            

        return fullDataset

    else:
        return torch.load(args.tensorData)
        #TODO: do something to load the dataset from file here

def trainNetworkOnData(dataset):
    """
    Takes in our TensorDataset, trains a ChessNet on it
    """
    numItems = len(dataset)
    train, valid = torch.utils.data.random_split(dataset, (int(numItems * 0.8), numItems - int((numItems * 0.8))))


    trainloader = torch.utils.data.DataLoader(train, batch_size=256)
    validloader = torch.utils.data.DataLoader(valid, batch_size=256)

    pdb.set_trace()
    model = ChessNet()

    #TODO: actually train


    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outData", help="File to which to write out loaded and parsed data",
                        type=argparse.FileType("wb"), nargs="?")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-p", "--pgnFiles", help="Filename(s) of pgn games to load",
                        type=argparse.FileType("r"), nargs="+")
    group.add_argument("pickledFile", help="Filename of picked alread-parsed pgn file to load",
                        type = argparse.FileType("r"), nargs="?")
    group.add_argument("-t", "--tensorData", help="Filename of file containing saved tensor data of a dataset",
                        type = argparse.FileType("rb"), nargs="?")

                        

    args = parser.parse_args()

    data = getTensorData(args)

    if args.outData != None:
        torch.save(data, args.outData)

    trainedModel = trainNetworkOnData(data)

    """
    myState = GameState.getInitialState()
    myMove  = Move("b2", "b4")

    moveTensor  = moveToTensor(myMove)
    stateTensor = gameStateToTensor(myState)
    logging.info(stateTensor)
    logging.info("stateTensor shape: (%s, %s, %s)" % (stateTensor.shape))
    logging.info(moveTensor)
    logging.info("moveTensor shape: (%s)" % (moveTensor.shape))
    """



if __name__ == "__main__":
    from log import setupLogging
    setupLogging()
    main()

