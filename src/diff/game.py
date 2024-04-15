import numpy as np

from .dealer import DiffDealer
from .player import DiffPlayer
from .judger import DiffJudger
from .round import DiffRound

class DiffGame:
    def __init__(self, n_players: int, rounds: int):
        self.np_random = np.random.RandomState()
        self.n_players = n_players
        self.rounds = rounds
        self.round = 0
        self.dealer = DiffDealer(self.np_random, self.n_players)
        self.players = [DiffPlayer(i, self.np_random) for i in range(self.n_players)]
        self.judger = DiffJudger()
        self.current_round = DiffRound(n_players, 0, self.dealer)

    def configure(self):
        pass

    def init_game(self):
        self.round = 0
