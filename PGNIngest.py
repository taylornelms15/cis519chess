"""
@file   PGNIngest.py
@author Taylor Nelms

Scripts to take in some PGN games and transform them into
ML-compatible game state representations
"""

import sys
import os
import re
import logging
import torch
import argparse

from log import setupLogging
from GameState import GameState, Move, Castle, Turn, BadGameParseException
from BitBoard import PieceType

import pdb

#Thought: initially, just mess around with the checkmate games?
ENDING_TYPES = {'White checkmated'          : -1,
                'Black checkmated'          : 1,
                'Game drawn by stalemate'   : 0,
                'Game drawn by repetition'  : 0}


#Want pairings of state->move
#Don't care about rewards at this point; imitation learning should contain those


def processPgnHalfmove(move, state):
    """
    Halfmove format:
    (Piece)(rank)(file)(x)[endloc](promotion)(check(m))
    """
    if re.match("O-O[^-]*", move):
        if state.getTurn() == Turn.WHITE:
            return Move.constructCastle(Castle.WKING)
        else:
            return Move.constructCastle(Castle.BKING)
    elif re.match("O-O-O[^-]*", move):
        if state.getTurn() == Turn.WHITE:
            return Move.constructCastle(Castle.WQUEEN)
        else:
            return Move.constructCastle(Castle.BQUEEN)

    #regstr = "([QKNRB]?)([a-h]?)([1-8]?)x?([a-h][1-8])((?=QNRB)?)[+#]?"
    regstr = "([QKNRB]?)([a-h]?)([1-8]?)x?([a-h][1-8])=?([QNRB]?)[+#]?"
    match = re.match(regstr, move)
    if match == None:
        logging.error(move)
    Piece, File, Rank, Endloc, Promotion = match.groups()

    return Move.constructFromPgnHalfmove(state, Piece, Rank, File, Endloc, Promotion)

def moveListFromGameLine(gLine):
    """
    This will actually need to keep track of the game state, because it affects how the moves get recorded
    """

    turnRegex = "\s*\d+\.\s"
    wholeTurns = re.split(turnRegex, gLine)[1:]#get rid of leading ''

    #logging.info(gLine)

    movePairs = []

    gameState = GameState.getInitialState()
    for turnString in wholeTurns:

        halfMoves = re.split(' ', turnString)
        mW = halfMoves[0]
        mB = None if len(halfMoves) < 2 else halfMoves[1]

        moveW = processPgnHalfmove(mW, gameState)
        movePairs.append( (gameState, moveW) )     
        gameState = moveW.apply(gameState)

        if mB:
            moveB = processPgnHalfmove(mB, gameState)
            movePairs.append( (gameState, moveB) )     
            gameState = moveB.apply(gameState)
            
    return movePairs

def processGameLine(line):
    regstr = "(.*?)\s+{(.*?)}\s+(.*)"
    game, reason, record = re.match(regstr, line).groups()

    if reason not in ENDING_TYPES:
        #for simplicity, ignoring games that did not end in a forced draw or a checkmate
        return None

    try:
        moveList = moveListFromGameLine(game)
    except BadGameParseException as e:
        return None

    return moveList
    #return ENDING_TYPES[reason]



def getNextGameLine(pgnFile):
    while(True):
        nextLine = pgnFile.readline()

        if nextLine == None or nextLine == '':
            pgnFile.close()
            return None

        if nextLine.startswith("1."):
            return nextLine.strip()

def parseAllLinesInFile(pgnFile):
    retval = []
    while True:
        nextGameLine = getNextGameLine(pgnFile)
        if nextGameLine == None:
            break
        result = processGameLine(nextGameLine)
        if result is not None:
            retval.append(result)
    return retval

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pgnFiles", help="Filename(s) of pgn games to load",
                        type=argparse.FileType("r"), nargs="+")
    args = parser.parse_args()

    fileNames = args.pgnFiles

    totalGames = 0
    rewardW = 0
    rewardB = 0
    reward0 = 0

    for fileName in args.pgnFiles:
        while(True):
            nextGameLine = getNextGameLine(fileName)

            if nextGameLine == None:
                break
            else:
                result = processGameLine(nextGameLine)
                totalGames += 1
                if result != None:
                    if result == 1:
                        rewardW += 1
                    elif result == -1:
                        rewardB += 1
                    else:
                        reward0 += 1

    logging.info("Total games %s, white %s, black %s, draw %s" % (totalGames, rewardW, rewardB, reward0))



if __name__ == "__main__":
    setupLogging()
    main()








