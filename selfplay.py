import os
from Supervised import SupervisedChess as SL
import argparse
import chess
from copy import deepcopy
from GameState import Turn, Castle, GameState, Move
from BitBoard import BitBoard, PieceType, BitBoardsFromFenString, FENParseString, S2I, I2S, Occupier, PIECELABELS
# from mcts1 import Node
from mcts_zero import MCTS


class SelfPlay:
    def __init__(self):
        self.n_games_per_iteration = 100
        self.model = SL(savedModel="modelBig.ptm")
        # MCTS arguments
        self.c_puct = 0.1
        self.mcts = MCTS(self.model, c_puct=self.c_puct)

    def mcts_one_match(self):
        new_state = chess.Board()
        model = self.model
        new_node = MCTS.Node(self.mcts, parent=None, state=new_state, move=None)
        examples = []

        while True:
            actual_state = new_state
            actual_node = new_node
            for s in range(self.mcts_simulations_per_state):
                self.mcts.search(actual_node)

            examples.append([actual_node.move, self.mcts.P[as_fen(actual_state)], None])
            best_move = actual_node.select_best_child().move

            new_state = deepcopy(actual_state)
            new_state.push(best_move)
            new_node = MCTS.Node(self.mcts, parent=actual_node, state=new_state, move=best_move)

            if new_state.is_game_over():
                examples = self.set_winner(examples, new_state.result())
                return examples