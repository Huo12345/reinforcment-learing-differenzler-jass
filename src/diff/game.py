from copy import deepcopy

import numpy as np

from rlcard.games.base import Card

from .dealer import DiffDealer
from .player import DiffPlayer
from .judger import DiffJudger
from .round import DiffRound
from .prediction import PredictionStrategy, RandomPredictionStrategy


class DiffGame:
    def __init__(self, allow_step_back=False, n_players=4) -> None:
        self.allow_step_back = allow_step_back
        if self.allow_step_back:
            self.history: list[tuple[int, DiffRound, list[DiffPlayer]]] = []
        self.np_random = np.random.RandomState()
        self.n_players = n_players
        self.rounds = 9
        self.round = 0
        self.dealer = DiffDealer(self.np_random, self.n_players)
        self.players = [DiffPlayer(i) for i in range(self.n_players)]
        self.judger = DiffJudger()
        self.current_round = DiffRound(self.n_players, 0, self.dealer)
        self.prediction_strategy: PredictionStrategy = RandomPredictionStrategy(self.np_random)

    def configure(self, game_config: dict) -> None:
        if game_config['players'] is not None:
            self.n_players = game_config['players']
            self.dealer = DiffDealer(self.np_random, self.n_players)
            self.players = [DiffPlayer(i) for i in range(self.n_players)]
            self.current_round = DiffRound(self.n_players, 0, self.dealer)
        if game_config['rounds'] is not None:
            self.rounds = game_config['rounds']
        if game_config['prediction_strategy'] is not None:
            self.prediction_strategy = game_config['prediction_strategy']

    def init_game(self) -> tuple[dict, int]:
        if self.allow_step_back:
            self.history = []
        self.round = 0
        for player in self.players:
            player.score = 0
        self._start_new_round(self.players)
        return self.get_state(self.current_round.current_player), self.current_round.current_player

    def step(self, action: str) -> tuple[dict, int]:
        self._add_state_to_history()

        action = Card(action[0], action[1])

        a = self.players[self.current_round.current_player].hand.index(action)
        self.current_round.proceed_round(self.players, a)
        if self.current_round.is_over():
            self._complete_round(self.players)
            if not self.is_over():
                self._start_new_round(self.players)
        return self.get_state(self.current_round.current_player), self.current_round.current_player

    def _start_new_round(self, players: list[DiffPlayer]) -> None:
        first_player = self.round % self.n_players
        self.current_round = DiffRound(self.n_players, first_player, self.dealer)
        self.current_round.deal_cards(players)
        for i in range(self.n_players):
            player = first_player + i % self.n_players
            state = self.get_state(self.current_round.current_player)
            prediction = self.prediction_strategy.get_prediction(player, state)
            self.current_round.make_prediction(players, prediction)

    def _complete_round(self, players: list[DiffPlayer]) -> None:
        self.round += 1
        for i in range(len(players)):
            player = players[i]
            self.prediction_strategy.provide_feedback(i, player.prediction, player.round_score)
            self.judger.score_player(player)

    def get_state(self, player: int) -> dict:
        return {
            "current_round": self.current_round.get_state(self.players, player),
            "player_scores": [p.score for p in self.players]
        }

    def get_full_state(self) -> dict:
        return {
            "current_round": self.current_round.get_full_state(self.players),
            "player_scores": [p.score for p in self.players]
        }

    def is_over(self) -> bool:
        return self.round >= self.rounds

    def _add_state_to_history(self) -> None:
        if self.allow_step_back:
            self.history.append((self.round, deepcopy(self.current_round), deepcopy(self.players)))

    def step_back(self) -> bool:
        if not self.allow_step_back or len(self.history) == 0:
            return False
        self.round, self.current_round, self.players = self.history.pop()
        return True

    def get_num_players(self) -> int:
        return self.n_players

    def get_num_actions(self) -> int:
        return len(self.dealer.deck)

    def get_player_id(self) -> int:
        return self.current_round.current_player

    def get_legal_actions(self) -> list[Card]:
        return self.current_round.get_legal_actions(self.players, self.current_round.current_player)

    def get_payoffs(self) -> list[float]:
        max_pts = 157 * self.round
        return [-1 * (p.score / max_pts) for p in self.players]
