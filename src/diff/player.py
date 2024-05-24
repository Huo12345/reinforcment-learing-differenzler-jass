import numpy as np


class DiffPlayer:

    def __init__(self, player_id: int) -> None:
        self.player_id = player_id
        self.hand = []
        self.score = 0
        self.prediction = 0
        self.round_score = 0

    def get_player_id(self):
        return self.player_id

    def get_state(self):
        return {
            "id": self.player_id,
            "hand": [c.get_index() for c in self.hand],
            "score": self.score,
            "prediction": self.prediction,
        }
