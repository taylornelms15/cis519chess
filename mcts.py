import logging
import numpy as np


from log import setupLogging

DEFAULT_POLICY_WEIGHT = 1e-2
EXPLORE_CONSTANT = 1e-1

class Edge(object):
    def __init__(self, parent, move):
        self.parent     = parent #the parent node game state
        self.move       = move   #descriptor of the chess move to take
        self.target     = None   #the game state this move would lead to



class Node(object):
    
    # Initialization

    def __init__(self, gameState):
        self.gameState      = gameState
        self.simReward      = 0.0
        self.numVisits      = 0
        self.edges          = []
        self.policyWeights  = [] #if we've trained a policy network to start from, its weights go here

        self.initEdges()

    def initEdges(self):
        """
        For the game state, figures out the possible moves, and creates edges from it
        Also creates empty nodes for the next level down
        """
        moveList    = self.gameState.getPossibleMoves()
        self.edges  = [Edge(self, x) for x in moveList]

        for i, edge in enumerate(self.edges):
            resultGameState = moveList[i].apply(self.gameState)
            edge.target = Node(resultGameState)

        self.policyWeights = [DEFAULT_POLICY_WEIGHT for x in self.edges]

    # Accessors

    def getNumVisits(self):
        return self.numVisits

    # Calculations

    def getExploitationTerm(self):
        """
        Q(v)/N(v)
        """
        if self.numVisits == 0:
            return 0
        else:
            return self.simReward / self.numVisits

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
            nextState = edge.target
            uct = nextState.getExploitationTerm() + EXPLORE_CONSTANT * self.policyWeights[i] * nextState.getExplorationTerm(self)
            uctArray[i] = uct

        return np.argmax(uctArray)

        


    





def main():
    logging.info("Running main function in mcts")




if __name__ == "__main__":
    setupLogging()
    main()
