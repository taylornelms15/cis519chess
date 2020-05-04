import math
from collections import defaultdict
from GameState import GameState
import chess
# from ZeroChess.utils import as_fen, as_board, process_nnet_output

# Attribution: https://github.com/XLekunberri/ZeroChess/blob/f701dd98a81f449aeb438b2d33b3a39a7680cd4a/ZeroChess/mcts.py#L6

class MCTS:
    def __init__(self, model, c_puct=0):
        self.nodes = {}
        self.model = model
        self.c_puct = c_puct

        self.Q = defaultdict(lambda: 0)
        self.N = defaultdict(lambda: 0)
        self.P = defaultdict(lambda: 0)

    def add_node(self, node):
        self.nodes[(node.parent, node.state, node.move)] = node
        for move in node.unexplored_children:
            node.add_child(move)
            self.Q[(node, move)] = 0
            self.N[(node, move)] = 0

    def search(self, node):
        # If the game is over, return the result
        fen_str = node.gameState.getFenString()
        board = chess.Board(fen=fen_str)
        if board.is_game_over():
            if board.result() == '1/2-1/2':
                return 0
            else:
                return -1

        # If this is the new state, evaluate with the neural network and update (EXPANSION)
        if node not in self.nodes.values():
            self.add_node(node)

            # pi, v = self.nnet.predict(as_board(node.state))

            legal_moves = node.gameState.getPossibleMoves()

            prob_list = self.model.getMovePreferenceList(node.gameState, legal_moves)
            # pi, v, compressed_pi = process_nnet_output(pi, v, legal_moves)

            self.P[node.gameState] = prob_list
            for move, i in zip(legal_moves, range(len(legal_moves))):
                for n in range(len(prob_list)):
                    if(prob_list[n][0] == move):
                        self.P[(node.state, move)] = prob_list[n][0]

            return -prob_list[0][0]

        # SELECTION
        best_child = node.select_best_child()
        best_move = best_child.move

        # SIMULATION
        v = self.search(best_child)

        # BACKPROPAGATION
        actual_q = self.Q[(node.gameState, best_move)]
        actual_n = self.N[(node.gameState, best_move)]

        self.Q[(node.state, best_move)] = (actual_n * actual_q + v) / (actual_n + 1)
        self.N[(node.state, best_move)] += 1

        return -v

    class Node:
        def __init__(self, mcts_instance, parent=None, gameState=None, move=None):
            self.MCTS = mcts_instance

            self.parent = parent
            self.gameState = gameState
            self.move = move
            self.children = []
            self.unexplored_children = gameState.getPossibleMoves()

        def get_ucb(self, gameState, move):
            q = self.MCTS.Q[(gameState, move)]
            p = self.MCTS.P[(gameState, move)] * math.sqrt(sum(self.MCTS.N.values())) / (1 + self.MCTS.N[(gameState, move)])

            return q + self.MCTS.c_puct * p
        
        def select_best_child(self):
            return sorted(self.children, key=lambda child: self.get_ucb(self.gameState, child.move))[0]

        def add_child(self, child_move):
            # child_state = as_board(self.gameState, move=child_move)
            child_state = child_move.apply(self.gameState)
            child = MCTS.Node(mcts_instance=self.MCTS, parent=self, gameState=child_state, move=child_move)
            self.children.append(child)
            self.unexplored_children.remove(child_move)