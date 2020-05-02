"""
@file result.py

Checking how the model is going to perform.
"""

# Initialization

import os


class testModel:
    """
    Class to test the Model.
    """

    def __init__(self):
        """
        :param config: Config to use to control how evaluation should work
        """
        self.model_eval = self.load_model()

    def load_model(self):
        """
        Loads the best model from the standard directory.
        :return ChessModel: the model
        """
        # Load the model here.
        model = getModel()

        return model

    def play_game(self):
        """
        Load the mdoel and check if the model performs better and save the result.
        """

    def evaluate_model(self):
        """
        Given a model, evaluates it by playing a bunch of games against the current model.
        """


def play_game()


"""
    Plays a game against models cur and ng and reports the results.
    :param Config config: config for how to play the game
    :param ChessModel cur: should be the current model
    :param ChessModel ng: should be the next generation model
    :param bool current_white: whether cur should play white or black
    :return (float, ChessEnv, bool): the score for the ng model
        (0 for loss, .5 for draw, 1 for win), the env after the game is finished, and a bool
        which is true iff cur played as white in that game.
    """
