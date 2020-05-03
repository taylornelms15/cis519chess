import logging
import numpy as np
import random
import copy
import chess

from log import setupLogging
from GameState import Turn, Castle, GameState, Move
from BitBoard import BitBoard, PieceType, Occupier, BitBoardsFromFenString, I2S


DEFAULT_POLICY_WEIGHT = 1e-2
EXPLORE_CONSTANT = 1e-1

TERMINATION_CHANCE = 1e-3

#Matches each game state (hashable, unaffected by training) to a node in our game tree
STATE_TO_NODE_DICT = {}

class Edge(object):
    def __init__(self, parent, move):
        self.parent         = parent #the parent node game state
        self.move           = move   #descriptor of the chess move to take
        self.target         = None   #the game state this move would lead to
        self.targetState    = None
        self.boardState     = None



class Node(object):
    
    # Initialization

    def __init__(self, gameState, board):
        self.gameState      = gameState
        self.qReward      = 0.0
        self.numVisits      = 0
        self.isTerminal     = True
        self.edges          = []
        self.policyWeights  = [] #if we've trained a policy network to start from, its weights go here
        self.board          = board

        self.initEdges()

    def initEdges(self):
        """
        For the game state, figures out the possible moves, and creates edges from it
        Also creates empty nodes for the next level down
        """
        moveList    = self.gameState.getPossibleMoves(self.board)
        self.edges  = [Edge(self, x) for x in moveList]

        for i, edge in enumerate(self.edges):
            resultGameState = moveList[i].apply(self.gameState)
            edge.targetState = resultGameState
            edge.target = None #otherwise, will recursively create every possible game state from here
            
            # additions to MCTS tp save python-chess type board-state
            final_move = I2S(moveList[i].startLoc) + I2S(moveList[i].endLoc)
            final_move = chess.Move.from_uci(final_move)
            self.board.push(final_move)
            edge.boardState = copy.deepcopy(self.board)
            self.board.pop()
            

        self.policyWeights = [DEFAULT_POLICY_WEIGHT for x in self.edges]

    def hasEdgeTargets(self):
        if len(self.edges) == 0:
            return True
        if (self.edges[0].target == None):
            return False
        return True

    def fillInEdgeTargets(self):
        for edge in self.edges:
            if edge.targetState not in STATE_TO_NODE_DICT:
                STATE_TO_NODE_DICT[edge.targeState] =  Node(edge.targetState, edge.boardState)
            edge.target = STATE_TO_NODE_DICT[edge.targetState]

    # Accessors

    def getNumVisits(self):
        return self.numVisits

    def getQReward(self):
        return self.qReward

    # Calculations

    def getExploitationTerm(self):
        """
        Q(v)/N(v)
        """
        if self.numVisits == 0:
            return 0
        else:
            return self.qReward / self.numVisits

    def getExplorationTerm(self, parent):
        """
        sqrt(N(v)/(1 + N(vi)))
        """
        return math.sqrt(parent.getNumVisits / (1.0 + self.getNumVisits()))

    # Traversal

    def getEdgeToTraverse(self):
        """
        Returns index of edge to go down
        """

        uctArray = np.zeros((len(self.edges)))

        for i in range(len(self.edges)):
            edge = self.edges[i]
            nextNode = edge.target
            if nextNode.turn == Turn.WHITE:#flips reward valuation based on whose turn it is
                exploitTerm = nextNode.getExploitationTerm()
            else:
                exploitTerm = -nextNode.getExploitationTerm()
            uct = nextNode.getExploitationTerm() + EXPLORE_CONSTANT * self.policyWeights[i] * nextNode.getExplorationTerm(self)
            uctArray[i] = uct

        return np.argmax(uctArray)


    def evaluateSubtree(self):
        """
        Traverses the game tree starting at this node
        """
        if random.random() <= TERMINATION_CHANCE:
            return 0.0#pretend we didn't visit this node, call it a draw

        self.numVisits += 1
        if self.gameState.isCheckmate():
            self.isTerminal = True
            if self.gameState.turn == Turn.WHITE:
                self.qReward += 1.0
            else:
                self.qReward += -1.0
        else:
            if not self.hasEdgeTargets():
                self.fillInEdgeTargets()#lazily expanding these nodes
            
            edgeToVisit = self.getEdgeToTraverse()

            childReward = self.edges[edgeToVisit].target.evaluateSubtree()#RECURSIVE
            self.updateWithNewChildReward(edgeToVisit, childReward)#backprop


        return self.qReward
        
    def updateWithNewChildReward(self, childIndex, childReward):
        """
        Updates our reward based on what a child sub-tree came back with
        """
        self.qReward += childReward




exampleWhiteCheckmate = "k7/8/1R6/8/3N4/8/8/7K w"#FEN string for chess board state where Nb5 checkmates

def main():
    random.seed(0xbadbad)
    logging.info("Running main function in mcts")

    exampleBoards = BitBoardsFromFenString(exampleWhiteCheckmate)

    logging.info(exampleBoards)
    for board in exampleBoards:
        logging.info(board)


if __name__ == "__main__":
    setupLogging()
    main()
