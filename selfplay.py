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
        self.n_games_per_iteration = 200
        self.model = SL(savedModel="modelBig2016.ptm")
        # MCTS arguments
        self.c_puct = 0.1
        self.mcts_simulations_per_state = 100
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

            # I am just keeping track of the board state using python-chess
            if(i% 20 == 0):
                print(board.fen())
            i +=1

            print(i, best_move)

            final_move = I2S(best_move.startLoc) + I2S(best_move.endLoc)
            final_move = chess.Move.from_uci(final_move)
            board.push(final_move)

            new_state = deepcopy(actual_state)
            new_state = best_move.apply(new_state)
            new_node = MCTS.Node(self.mcts, parent=actual_node, gameState=new_state, move=best_move)

            # Debug for end game
            if board.result() != "*":
                if self.winner is None:
                    self.result = board.result()
                    
                if self.result == '1-0':
                    print("White winner.")
                elif self.result == '0-1':
                    print("Black winner.")
                else:
                    print("Draw.. Chicken dinner not served..")

                return board

    def save_pgn(self, board, n_game, name, n_iter, white="WHITE", black="BLACK"):
        game = chess.pgn.Game()
        game.headers["Event"] = 'Self-Play'
        game.headers["Site"] = 'None'
        game.headers["Date"] = "123"
        game.headers["Round"] = n_game
        game.headers["White"] = white
        game.headers["Black"] = black
        game.headers["WhiteElo"] = "NA"
        game.headers["BlackElo"] = "NA"
        game.headers["WhiteRD"] = "NA"
        game.headers["BlackRD"] = "NA"
        game.headers["WhiteIsComp"] = "NA"
        game.headers["TimeControl"] = "na"
        game.headers["Date"] = board.result()
        game.headers["Time"] = "na"
        game.headers["WhiteClock"] = board.result()
        game.headers["BlackClock"] = "na"
        game.headers["ECO"] = "na"
        game.headers["Plycount"] = "na"
        game.headers["Result"] = board.result()

        moves = [move for move in board.move_stack]
        node = game.add_variation(moves.pop(0))
        n_move = 1
        for move in moves:
            node = node.add_variation(move)
            n_move += 1

        # Update this path
        path = 'C:\\Personal\\Masters Study Material\\CIS 519\\Project\\cis519chess\\games'
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path + '\\self-play.pgn'.format(n_iter=n_iter), 'a') as pgn:
            pgn.write(game.accept(chess.pgn.StringExporter()))
            pgn.write('\n\n')

def main():
    play_match = SelfPlay()
    for i in range(200):
        board = play_match.mcts_one_match()
        play_match.save_pgn(board=board, n_game=i, name="MCTS", n_iter=i)

if __name__ == "__main__":
    from log import setupLogging
    setupLogging()
    main()
