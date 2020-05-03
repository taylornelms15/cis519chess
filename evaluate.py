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


class testModel:
    """
    Class to test the Model.
    """

    def __init__(self):
        """
        :param config: Config to use to control how evaluation should work
        """
        parser = argparse.ArgumentParser()
        self.chessClassW = SL(savedModel="modelBig.ptm")
        self.chessClassB = SL(savedModel="modelBig.ptm")
        self.myState = GameState.getInitialState()
        self.board = chess.Board()
        self.mcts_node = Node(self.myState, self.board)

    def reset(self):
        self.board = None
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

    def play_game(self):
        """
            Load the model and check if the model performs better and save the result.
            """
        self.reset()
        i = 0

        while self.winner is None:
            if i % 2 == 0:
                legal_move = self.legal_move()
                move = self.chessClassW.getMovePreference(
                    self.myState, legal_move)

                self.log_move(move)
                self.myState = move.apply(self.myState)

            else:

                legal_move = self.legal_move()
                move = self.chessClassB.getMovePreference(
                    self.myState, legal_move)

                self.log_move(move)
                self.myState = move.apply(self.myState)

            self.num_halfmoves += 1

            if self.board.result(claim_draw=True) != "*":
                if self.winner is None:
                    self.result = self.board.result(claim_draw=True)
                if self.result == '1-0':
                    self.winner = 1
                elif self.result == '0-1':
                    self.winner = -1
                else:
                    self.winner = 0.5

            print(move)
            print()
            i = i + 1

        print(self.result)

        # def evaluate_model(self):
        #     """
        #     Given a model, evaluates it by playing a bunch of games against the current model.
        #     """


def main():
    """
    Plays a game against model.
    """
    model = testModel()
    model.play_game()


if __name__ == "__main__":
    main()
