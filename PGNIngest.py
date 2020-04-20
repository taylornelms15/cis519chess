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

import pdb

#Thought: initially, just mess around with the checkmate games?
ENDING_TYPES = {'White checkmated'          : -1,
                'Black checkmated'          : 1,
                'Game drawn by stalemate'   : 0,
                'Game drawn by repetition'  : 0}


#Want pairings of state->move
#Don't care about rewards at this point; imitation learning should contain those

def moveListFromGameLine(gLine):
    pass

def processGameLine(line):
    regstr = "(.*?)\s+{(.*?)}\s+(.*)"
    game, reason, record = re.match(regstr, line).groups()

    if reason not in ENDING_TYPES:
        #for simplicity, ignoring games that did not end in a forced draw or a checkmate
        return None

    moveList = moveListFromGameLine(game)

    return ENDING_TYPES[reason]



def getNextGameLine(pgnFile):
    while(True):
        nextLine = pgnFile.readline()

        if nextLine == None or nextLine == '':
            pgnFile.close()
            return None

        if nextLine.startswith("1."):
            return nextLine.strip()

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








