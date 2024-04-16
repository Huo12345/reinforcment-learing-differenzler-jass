import numpy as np

from rlcard.games.base import Card

from .dealer import DiffDealer
from .player import DiffPlayer
from .judger import DiffJudger
from .round import DiffRound
from .prediction import PredictionStrategy


class DiffGame:
    def __init__(self, n_players: int, rounds: int, prediction_strategy: PredictionStrategy) -> None:
        self.np_random = np.random.RandomState()
        self.n_players = n_players
        self.rounds = rounds
        self.round = 0
        self.dealer = DiffDealer(self.np_random, self.n_players)
        self.players = [DiffPlayer(i, self.np_random) for i in range(self.n_players)]
        self.judger = DiffJudger()
        self.current_round = DiffRound(n_players, 0, self.dealer)
        self.prediction_strategy = prediction_strategy

    def configure(self):
        pass

    def init_game(self):
        self.round = 0
        for player in self.players:
            player.score = 0
        self._start_new_round(self.players)

    def step(self, action: Card) -> type[dict, int]:
        a = self.players[self.current_round.current_player].hand.index(action)
        self.current_round.proceed_round(self.players, a)
        if self.current_round.is_over():
            self._complete_round(self.players)
            if not self.is_over():
                self._start_new_round(self.players)
        return self.get_state(self.players), self.current_round.current_player

    def _start_new_round(self, players: list[DiffPlayer]):
        self.round += 1
        first_player = self.round % self.n_players
        self.current_round = DiffRound(self.n_players, first_player, self.dealer)
        self.current_round.deal_cards(players)
        for i in range(self.n_players):
            player = first_player + i % self.n_players
            prediction = self.prediction_strategy.get_prediction(player, self.get_state(players))
            self.current_round.make_prediction(players, prediction)

    def _complete_round(self, players: list[DiffPlayer]):
        for player in players:
            self.judger.score_player(player)

    def get_state(self, players: list[DiffPlayer]) -> dict:
        return {
            "current_round": self.current_round.get_state(players),
            "player_scores": [p.score for p in players]
        }

    def is_over(self) -> bool:
        return self.round > self.rounds
