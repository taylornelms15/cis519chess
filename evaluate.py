"""
@file evaluate.py

Checking how the model is going to perform.
"""

# Initialization

import os
from Supervised import SupervisedChess as SL
import argparse
import chess
from GameState import Turn, Castle, GameState, Move
from BitBoard import BitBoard, PieceType, BitBoardsFromFenString, FENParseString, S2I, I2S, Occupier, PIECELABELS
from mcts1 import Node
import chess.pgn


class testModel:
    """
    Class to test the Model.
    """

    def __init__(self):
        """
        :param config: Config to use to control how evaluation should work
        """
        self.chessClassW = SL(savedModel="modelBig2016.ptm")
        self.chessClassB = SL(savedModel="modelBig2016.ptm")
        self.myState = GameState.getInitialState()
        self.board = chess.Board()
        self.mcts_node = Node(self.myState, self.board, self.chessClassB)
        self.num_games = 4
        self.max_game_length = 1000

    def reset(self):
        self.num_halfmoves = 0
        self.winner = None  # type: Winner
        self.turn = None
        self.resigned = False
        self.result = None
        self.board = chess.Board()
        self.myState = GameState.getInitialState()
        self.legalMoves = [
            Move('a2', 'a3'),
            Move('a2', 'a4'),
            Move('b2', 'b3'),
            Move('b2', 'b4'),
            Move('c2', 'c3'),
            Move('c2', 'c4'),
            Move('d2', 'd3'),
            Move('d2', 'd4'),
            Move('e2', 'e3'),
            Move('e2', 'e4'),
            Move('f2', 'f3'),
            Move('f2', 'f4'),
            Move('g2', 'g3'),
            Move('g2', 'g4'),
            Move('h2', 'h3'),
            Move('h2', 'h4'),
            Move('b1', 'a3'),
            Move('b1', 'c3'),
            Move('g1', 'f3'),
            Move('g1', 'h3'),
        ]

    def legal_move(self):
        """
        Function to get all the legal moves.
        """
        # used python-chess to get all legal moves.
        legal_moves = []

        for move in self.board.legal_moves:
            move_str = move.uci()
            start_loc = move_str[:2]
            end_loc = move_str[2:]

            # Casting the move returned by the python-chess to our implementation.
            my_move = Move(start_loc, end_loc)
            legal_moves.append(my_move)

        return legal_moves

    def log_move(self, move):
        # change the state of python-chess board in parallel with our gameState.
        final_move = I2S(move.startLoc) + I2S(move.endLoc)
        final_move = chess.Move.from_uci(final_move)
        self.board.push(final_move)

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

    def play_game(self):
        """
            Load the model and check if the model performs better and save the result.
            """
        self.reset()
        i = 0
        j = 0
        self.Player1Score = 0.0
        self.Player2Score = 0.0
        for j in range(self.num_games):

            if(j % 2 == 0):
                Player1 = self.chessClassW
                Player2 = self.chessClassB
            else:
                Player2 = self.chessClassW
                Player1 = self.chessClassB

            while self.winner is None:
                if i % 2 == 0:
                    legal_move = self.legal_move()
                    move = Player1.getMovePreference(
                        self.myState, legal_move)

                    self.log_move(move)
                    self.myState = move.apply(self.myState)

                else:

                    legal_move = self.legal_move()
                    move = Player2.getMovePreference(
                        self.myState, legal_move)

                    self.log_move(move)
                    self.myState = move.apply(self.myState)

                self.num_halfmoves += 1

                if self.num_halfmoves >= self.max_game_length:
                    break

                if self.board.result() != "*":
                    if self.winner is None:
                        self.result = self.board.result()
                    if self.result == '1-0':
                        if(j % 2 == 0):
                            self.Player1Score += 1
                            self.winner = 1
                        else:
                            self.Player2Score += 1
                            self.winner = 1

                    elif self.result == '0-1':
                        if(j % 2 == 0):
                            self.Player1Score += 1
                            self.winner = 1
                        else:
                            self.Player2Score += 1
                            self.winner = 1
                    else:
                        self.winner = 0.5
                        self.Player1Score += 0.5
                        self.Player2Score += 0.5
                i = i + 1

            print(self.result)
            self.save_pgn(board=self.board, n_game=j, name="Harsh", n_iter=i)
            self.winner = None
            self.num_halfmoves = 0
            i = 0
            self.reset()

            print(self.Player1Score)
            print(self.Player2Score)


def main():
    """
    Plays a game against model.
    """
    model = testModel()
    model.play_game()


if __name__ == "__main__":
    main()
