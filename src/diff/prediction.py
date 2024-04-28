import numpy as np


class PredictionStrategy:
    def get_prediction(self, player: int, game_state: dict) -> int:
        pass

    def provide_feedback(self, player: int, prediction_made: int, points_reached: int) -> None:
        pass


class CompositePredictionStrategy(PredictionStrategy):

    def __int__(self, player_strategies: list[PredictionStrategy]):
        self.player_strategies = player_strategies

    def get_prediction(self, player: int, game_state: dict) -> int:
        return self.player_strategies[player].get_prediction(player, game_state)

    def provide_feedback(self, player: int, prediction_made: int, points_reached: int) -> None:
        self.player_strategies[player].provide_feedback(player, prediction_made, points_reached)


class RandomPredictionStrategy(PredictionStrategy):

    def __init__(self, np_random: np.random) -> None:
        self.np_random = np_random

    def get_prediction(self, player: int, game_state: dict) -> int:
        return self.np_random.randint(158)


class FixedPredictionStrategy(PredictionStrategy):

    def __init__(self, prediction: int) -> None:
        self.prediction = prediction

    def get_prediction(self, player: int, game_state: dict) -> int:
        return self.prediction
