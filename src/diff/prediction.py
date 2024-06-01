import numpy as np


class PredictionStrategy:
    """
    Interface that allows implementations on how to obtain a prediction for a hand in a round.
    """
    def get_prediction(self, player: int, game_state: dict) -> int:
        """
        Calculates a prediction for a hand.

        :param player: Player that needs to predict their score
        :param game_state: Observation of the game from the current player
        :return: Score between 0 and 157 indicating how many points the player wants to make this round
        """
        pass

    def provide_feedback(self, player: int, prediction_made: int, points_reached: int) -> None:
        """
        Provides feedback on the points effectively reached for learning based approaches.

        :param player: The player that made the prediction.
        :param prediction_made: The prediction the player made.
        :param points_reached: Effective points reached by the player this round.
        """
        pass


class CompositePredictionStrategy(PredictionStrategy):
    """
    A helper class that allows to define a strategy per player. Takes a list of strategies and delegates the decision
    for each player to the strategy at the index corresponding to the player id.
    """

    def __int__(self, player_strategies: list[PredictionStrategy]) -> None:
        """
        Initialises the delegating strategy.

        :param player_strategies: List of strategies for the different players
        """
        self.player_strategies = player_strategies

    def get_prediction(self, player: int, game_state: dict) -> int:
        """
        Delegates the prediction to the strategy defined for the player.

        :param player: Player that needs to predict their score
        :param game_state: Observation of the game from the current player
        :return: Score between 0 and 157 indicating how many points the player wants to make this round
        """
        return self.player_strategies[player].get_prediction(player, game_state)

    def provide_feedback(self, player: int, prediction_made: int, points_reached: int) -> None:
        """
        Forwards the feedback to the strategy that made the decision for the given player.

        :param player: The player that made the prediction.
        :param prediction_made: The prediction the player made.
        :param points_reached: Effective points reached by the player this round.
        """
        self.player_strategies[player].provide_feedback(player, prediction_made, points_reached)


class RandomPredictionStrategy(PredictionStrategy):
    """
    Basic strategy that randomly chooses a number between 0 and 157.
    """

    def __init__(self, np_random: np.random) -> None:
        """
        Initializes the random strategy.

        :param np_random: Random state for reproducibility.
        """
        self.np_random = np_random

    def get_prediction(self, player: int, game_state: dict) -> int:
        """
        Chooses a random number uniformly between 0 and 157.

        :param player: Player that needs to predict their score, ignored.
        :param game_state: Observation of the game from the current player, ignored.
        :return: A random value between 0 and 157.
        """
        return self.np_random.randint(158)


class FixedPredictionStrategy(PredictionStrategy):
    """
    Basic strategy that always predicts the same score.
    """

    def __init__(self, prediction: int) -> None:
        """
        Initializes the fixed strategy.

        :param prediction: Number of points that will always be predicted.
        """
        self.prediction = prediction

    def get_prediction(self, player: int, game_state: dict) -> int:
        """
        Returns the fixed value for the prediction.

        :param player: Player that needs to predict their score, ignored.
        :param game_state: Observation of the game from the current player, ignored.
        :return: The fixed value for the prediction.
        """
        return self.prediction
