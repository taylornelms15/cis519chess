import logging
from log import setupLogging




class Edge(object):
    def __init__(self, parent, move):
        self.parent     = parent #the parent node game state
        self.move       = move   #descriptor of the chess move to take
        self.target     = None   #the game state this move would lead to



class Node(object):
    
    def __init__(self, gameState):
        self.gameState  = gameState
        self.simReward  = 0.0
        self.numVisits  = 0
        self.edges      = []

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







def main():
    logging.info("Running main function in mcts")




if __name__ == "__main__":
    setupLogging()
    main()
