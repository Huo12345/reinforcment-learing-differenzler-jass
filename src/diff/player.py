import numpy as np


class DiffPlayer:

    def __init__(self, player_id: int, np_random: np.random) -> None:
        self.np_random = np_random
        self.player_id = player_id
        self.hand = []
        self.score = 0
        self.prediction = 0
        self.round_score = 0

    def get_player_id(self):
        return self.player_id