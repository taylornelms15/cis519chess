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
        self.mcts_simulations_per_state = 20
        self.mcts = MCTS(self.model, c_puct=self.c_puct)

        # Debug
        self.winner = None
        self.result = None

    def mcts_one_match(self):
        new_state = GameState.getInitialState()
        new_node = MCTS.Node(self.mcts, parent=None, gameState=new_state, move=None)
        examples = []

        # Debug
        i = 0   
        board = chess.Board()

        while True:

            actual_state = new_state
            actual_node = new_node

            # Running the exploration step of MCTS.
            for s in range(self.mcts_simulations_per_state):
                self.mcts.search(actual_node)

            examples.append([actual_node.move, self.mcts.P[actual_state], None])
            best_move = actual_node.select_best_child().move

            # Debug Start.
            # I am just keeping track of the board state using python-chess
            if(i% 20 == 0):
                print(board.fen())
            i +=1

            print(i, best_move)

            final_move = I2S(best_move.startLoc) + I2S(best_move.endLoc)
            final_move = chess.Move.from_uci(final_move)
            board.push(final_move)
            # Debug End

            new_state = deepcopy(actual_state)
            new_state = best_move.apply(new_state)
            new_node = MCTS.Node(self.mcts, parent=actual_node, gameState=new_state, move=best_move)

            # Debug for end game
            if board.result() != "*":
                if self.winner is None:
                    self.result = board.result()
                if self.result == '1-0':
                    print("chicken dinner ww")
                    break
                elif self.result == '0-1':
                    print("chicken dinner bb")
                    break
                else:
                    print("chicken dinner draw.")
                    break
            
            # Break if game is Over!!
            if new_state.isGameOver() or board.is_game_over():
                # examples = self.set_winner(examples, new_state.result())
                print("chicken dinner!")
                return examples # Need to check what exactly to return.

def main():
    play_match = SelfPlay()
    play_match.mcts_one_match()

if __name__ == "__main__":
    from log import setupLogging
    setupLogging()
    main()
