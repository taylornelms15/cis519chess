import logging




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
        moveList = getPossibleMoves(self.gameState)
        self.edges = [Edge(self, x) for x in moveList]

        for i, edge in enumerate(self.edges):
            resultGameState = applyMove(self.gameState, moveList[i])
            edge.target = Node(resultGameState)






def setupLogging():
    logging.basicConfig(level=logging.INFO, format='%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

def main():
    setupLogging()
    logging.info("Running main function in mcts")




if __name__ == "__main__":
    main()
