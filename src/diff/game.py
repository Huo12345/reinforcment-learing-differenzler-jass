from copy import deepcopy

import numpy as np

from .dealer import DiffDealer
from .player import DiffPlayer
from .judger import DiffJudger
from .round import DiffRound
from .prediction import PredictionStrategy, RandomPredictionStrategy
from .utility import card_from_str


class DiffGame:
    """
    Represents a game played over multiple rounds. Contains a dealer, a judger, n players and the current round.
    """

    def __init__(self, allow_step_back: bool = False, n_players: int = 4) -> None:
        """
        Initializes the game.

        :param allow_step_back: If true allows a player to take a step back and restore the state before the last action
        taken.
        :param n_players: Number of players in the game.
        """
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
        self.reward_strategy = 'default'
        self.first_player_strategy = 'normal'

    def configure(self, game_config: dict) -> None:
        """
        Configures the current game. The configuration can contain the following entries:
        players [int]: Number of players between 3 and 5.
        rounds [int]: Then number of rounds to play.
        prediction_strategy [PredictionStrategy object]: Strategy to use for predicting the number of points in a round
            given a specific hand.
        reward_strategy [str]: Strategy to use for reward calculation. Allowed values {default, winner_takes_all,
            constant}. Default and constant return the difference between the prediction and the score normalized by the
            maximum possible points off and negated (between 0 and -1). Constant gives it at every step, default only at
            the end of the game. The winner_takes_all strategy returns 1 for the player with the fewest difference
            between prediction and score and if players are tied, they share the points.
        first_player_strategy [str]: How to choose the first player in the first round. Allowed values {default, random}

        :param game_config: Configuration of the game.
        """
        if 'players' in game_config and game_config['players'] is not None:
            self.n_players = game_config['players']
            self.dealer = DiffDealer(self.np_random, self.n_players)
            self.players = [DiffPlayer(i) for i in range(self.n_players)]
            self.current_round = DiffRound(self.n_players, 0, self.dealer)
        if 'rounds' in game_config and game_config['rounds'] is not None:
            self.rounds = game_config['rounds']
        if 'prediction_strategy' in game_config and game_config['prediction_strategy'] is not None:
            self.prediction_strategy = game_config['prediction_strategy']
        if 'reward_strategy' in game_config and game_config['reward_strategy'] is not None:
            self.reward_strategy = game_config['reward_strategy']
        if 'first_player_strategy' in game_config and game_config['first_player_strategy'] is not None:
            self.first_player_strategy = game_config['first_player_strategy']

    def init_game(self) -> tuple[dict, int]:
        """
        Resets the game to the initial state.

        :return: State of the game at the beginning.
        """
        if self.allow_step_back:
            self.history = []
        self.round = 0
        for player in self.players:
            player.score = 0
        self._start_new_round(self.players)
        return self.get_state(self.current_round.current_player), self.current_round.current_player

    def step(self, action: str) -> tuple[dict, int]:
        """
        Executes the action on the current players behalf.

        :param action: Action to execute: String representation of the card to be played.
        :return: The state after the action was executed
        """
        self._add_state_to_history()

        action = card_from_str(action)

        a = self.players[self.current_round.current_player].hand.index(action)
        self.current_round.proceed_round(self.players, a)
        if self.current_round.is_over():
            self._complete_round(self.players)
            if not self.is_over():
                self._start_new_round(self.players)
        return self.get_state(self.current_round.current_player), self.current_round.current_player

    def _start_new_round(self, players: list[DiffPlayer]) -> None:
        """
        Starts a new round.

        :param players: List of all playing players.
        """
        if self.round == 0:
            first_player = self.np_random.randint(0, 4) if self.first_player_strategy == 'random' else 0
        else:
            first_player = (self.current_round.first_player + 1) % self.n_players

        self.current_round = DiffRound(self.n_players, first_player, self.dealer)
        self.current_round.deal_cards(players)
        for i in range(self.n_players):
            player = (first_player + i) % self.n_players
            state = self.get_state(player)
            prediction = self.prediction_strategy.get_prediction(player, state)
            self.current_round.make_prediction(players, prediction)

    def _complete_round(self, players: list[DiffPlayer]) -> None:
        """
        Completes the current round.

        :param players: List of all playing players.
        """
        self.round += 1
        for i in range(len(players)):
            player = players[i]
            self.prediction_strategy.provide_feedback(i, player.prediction, player.round_score)
            self.judger.score_player(player)

    def get_state(self, player: int) -> dict:
        """
        Extracts the observation of the game for a given player.

        :param player: Id of the player to get the observation for.
        :return: Observation of the game.
        """
        return {
            "current_round": self.current_round.get_state(self.players, player),
            "player_scores": [p.score for p in self.players]
        }

    def get_full_state(self) -> dict:
        """
        Extracts the full state of the game.

        :return: Dict containing the full state of the game.
        """
        return {
            "current_round": self.current_round.get_full_state(self.players),
            "player_scores": [p.score for p in self.players]
        }

    def is_over(self) -> bool:
        """
        Returns true if the game is over.

        :return: True if the game is over, False otherwise.
        """
        return self.round >= self.rounds

    def _add_state_to_history(self) -> None:
        """
        Pushes the current state of the game to the history to allow step_back.
        """
        if self.allow_step_back:
            self.history.append((self.round, deepcopy(self.current_round), deepcopy(self.players)))

    def step_back(self) -> bool:
        """
        Restores the state before the last action taken. If a step back is not possible, returns False.

        :return: True if the previous state could be loaded, False otherwise.
        """
        if not self.allow_step_back or len(self.history) == 0:
            return False
        self.round, self.current_round, self.players = self.history.pop()
        return True

    def get_num_players(self) -> int:
        """
        Returns the number of players in the game.
        :return: Number of players in the game.
        """
        return self.n_players

    def get_num_actions(self) -> int:
        """
        Returns how many actions are possible in the game

        :return: Number of possible actions
        """
        return len(self.dealer.deck)

    def get_player_id(self) -> int:
        """
        Returns the player id of the current player.

        :return: Id of the current player.
        """
        return self.current_round.current_player

    def get_legal_actions(self) -> list[str]:
        """
        Finds the legal actions for the current player.

        :return: List of legal actions for the current player, represented as strings for the cards that can be played.
        """
        return self.current_round.get_legal_actions(self.players, self.current_round.current_player)

    def get_payoffs(self) -> list[float]:
        """
        Returns the payoffs for all players.

        :return: List of payoffs for all players.
        """
        scores = [p.score for p in self.players]
        if self.reward_strategy == 'winner_takes_all':
            return self._winner_takes_all_payoff(scores)
        if self.reward_strategy == 'constant':
            return self._constant_payoff(scores)
        return self._default_payoff(scores)

    def _winner_takes_all_payoff(self, scores: list[float]) -> list[float]:
        """
        Calculates the player scores using the winner takes all strategy.

        :param scores: Player scores (sum of the number of points the predictions were off in all previous rounds).
        :return: List of scores for all players.
        """
        if self.round == 0:
            return [0 for _ in range(self.n_players)]
        best = min(scores)
        results = [1 if score == best else 0 for score in scores]
        norm = sum(results)
        return [i / norm for i in results]

    def _default_payoff(self, scores: list[float]) -> list[float]:
        """
        Calculates the player scores using the default strategy.

        :param scores: Player scores (sum of the number of points the predictions were off in all previous rounds).
        :return: List of scores for all players.
        """
        if self.round == 0:
            return [0 for _ in range(self.n_players)]
        max_pts = 157. * self.round
        return [-1. * (score / max_pts) for score in scores]

    def _constant_payoff(self, scores: list[float]):
        """
        Calculates the player scores using the constant strategy.

        :param scores: Player scores (sum of the number of points the predictions were off in all previous rounds).
        :return: List of scores for all players.
        """
        if self.current_round is None or self.current_round.is_over():
            return self._default_payoff(scores)
        max_pts = 157. * (self.round + 1)
        addition = [abs(p.prediction - p.round_score) for p in self.players]
        return [-1. * (a + b) / max_pts for (a, b) in zip(scores, addition)]
