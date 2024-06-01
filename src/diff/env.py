import random
from collections import OrderedDict

import numpy as np
from numpy import ndarray
from rlcard.envs import Env

from .game import DiffGame
from .utility import get_full_deck, card_from_str, followed_suit, find_strong_cards, find_weak_cards

DEFAULT_GAME_CONFIG = {
    'players': 4,
    'rounds': 9,
    'state_representation': 'compressed',
    'allow_step_back': False,
    'seed': random.randint(1, 999999)
}


class DiffEnv(Env):
    """
    Environment class that allows RLCard to interact with the implementation of Differenzler.
    """

    def __init__(self, config: dict | None = None) -> None:
        """
        Initializes the environment based on a configuration. Uses the default configuration if none is passed.

        :param config: A dictionary containing the configuration of the environment.
        """
        self.name = 'differenzler'

        # Initializing config
        self.default_game_config = DEFAULT_GAME_CONFIG
        if config is None:
            config = DEFAULT_GAME_CONFIG
        super().__init__(config)

        # Initializing Game
        self.game = DiffGame()
        self.game.configure(config)

        # Preparing state representation
        self.state_rep = config['state_representation'] if 'state_representation' in config else 'default'
        if self.state_rep == 'compressed' or self.state_rep == 'enhanced_small':
            self.state_shape = [[4, 36] for _ in range(self.num_players)]
        elif self.state_rep == 'enhanced_large':
            self.state_shape = [[6, 36] for _ in range(self.num_players)]
        else:
            self.state_shape = [[42, 40] for _ in range(self.num_players)]

        # Setting action space size
        self.action_shape = [36 for _ in range(self.num_players)]

        # Creating utility variables
        self.reference_deck = get_full_deck()
        self.suits = ['S', 'H', 'D', 'C']

    def _extract_state(self, state: dict) -> dict:
        """
        Extracts the state of the game into a representation useful for RLCard.

        :param state: Raw state of the game
        :return: Enriched state for RLCard
        """

        # Creating observation vector representation
        if self.state_rep == 'compressed' or self.state_rep == 'enhanced_small':
            obs = self._parse_state_to_compressed_ndarray(state, self.state_rep == 'compressed')
        elif self.state_rep == 'enhanced_large':
            obs = self._parse_state_to_enhanced_ndarray(state)
        else:
            obs = self._parse_state_to_ndarray(state)

        # Creating enriched state
        return {
            'obs': obs,
            'legal_actions': self._get_legal_actions(),
            'raw_obs': state,
            'raw_legal_actions': state['current_round']['legal_moves'],
            'action_record': self.action_recorder
        }

    def _parse_state_to_compressed_ndarray(self, state: dict, enhanced: bool) -> ndarray:
        """
        Parses the state into a compressed version of the tensor containing all cards played, the hand of the player,
        the current pile, the trump suit, the scores and predictions of the current round. If the state is enhanced also
        includes which player has followed on which suit so far.

        :param state: Raw state of the game
        :param enhanced: Whether the state is enhanced or not.
        :return: An observation tensor of size [4, 36]
        """
        # Initializes observation vector
        obs = np.zeros((4, 36), dtype=int)
        current_round = state['current_round']

        # Parses the common parts of the compressed state
        self._parse_common_compressed_state(obs, current_round)

        # If the state is not enhanced, returns the observation vector without the followed suit information
        if not enhanced:
            return obs

        # Parses the followed suit information
        self._parse_followed_suit(obs, current_round)

        return obs

    def _parse_state_to_enhanced_ndarray(self, state: dict) -> ndarray:
        """
        Parses the state into a compressed version of the tensor containing all cards played, the hand of the player,
        the current pile, the trump suit, the scores and predictions of the current round, which player followed suit,
        which cards are strong (can only be taken by trump) and which cards are weak (have to be taken).

        :param state: Raw state of the game
        :return: An observation tensor of size [6, 36]
        """
        # Initializes observation vector
        obs = np.zeros((6, 36), dtype=int)
        current_round = state['current_round']

        # Parses the common parts of the compressed state
        self._parse_common_compressed_state(obs, current_round)

        # Parses the followed suit information
        self._parse_followed_suit(obs, current_round)

        # Encoding strong cards
        strong_cards = find_strong_cards(current_round['played_piles'],
                                         current_round['player']['hand'],
                                         current_round['trump'])
        for card in strong_cards:
            obs[4, self.reference_deck.index(card_from_str(card))] = 1

        # Encoding weak cards
        weak_cards = find_weak_cards(current_round['played_piles'],
                                     current_round['player']['hand'],
                                     current_round['trump'])
        for card in weak_cards:
            obs[4, self.reference_deck.index(card_from_str(card))] = 1

        return obs

    def _parse_common_compressed_state(self, obs: ndarray, current_round: dict) -> None:
        """
        Parses the parts of the compressed state that all representations share. Populates the obs tensor accordingly.

        :param obs: Observation tensor to write the values into, at least size [4, 36] expected
        :param current_round: The state of the current round of the game state
        """
        # Parsing history
        for pile in current_round['played_piles']:
            for card in pile['played_cards']:
                obs[0, self.reference_deck.index(card_from_str(card))] = 1

        # Parsing current pile
        for card in current_round['current_pile']['played_cards']:
            obs[1, self.reference_deck.index(card_from_str(card))] = 1

        # Parsing current hand
        for card in current_round['player']['hand']:
            obs[2, self.reference_deck.index(card_from_str(card))] = 1

        # Parsing scores and predictions
        obs[3, current_round['current_player']] = 1
        for i, pred in enumerate(current_round['predictions']):
            obs[3, i + 4] = pred if pred is not None else 0
        for i, score in enumerate(current_round['round_scores']):
            obs[3, i + 8] = score

        # Encoding trump
        obs[3, 12 + self.suits.index(current_round['trump'])] = 1

    def _parse_followed_suit(self, obs: ndarray, current_round: dict) -> None:
        """
        Parses which player has followed suit so far. Updates the obs tensor accordingly.

        :param obs: Observation tensor to write the values into, at least size [4, 36] expected
        :param current_round: The state of the current round of the game state
        """
        # Encoding followed suit
        followed = followed_suit(current_round['played_piles'], self.num_players, current_round['trump'])
        obs[3, 16:32] = [p for s in followed for p in s]

    def _parse_state_to_ndarray(self, state: dict) -> ndarray:
        """
        Parses the state of the came into an uncompressed vector containing the full history played, the hand of the
        player, the current pile, the trump suit, the scores and predictions of the current round.

        :param state: Raw state of the game
        :return: An observation tensor of size [42, 40]
        """
        obs = np.zeros((42, 40), dtype=int)

        # Paring history
        for i in range(len(state['current_round']['played_piles'])):
            pile = state['current_round']['played_piles'][i]
            round_initiator = pile['first_player']
            played_cards = pile['played_cards']
            for j in range(len(played_cards)):
                obs[i * 4 + j, round_initiator] = 1
                obs[i * 4 + j, 4 + self.reference_deck.index(card_from_str(played_cards[j]))] = 1

        # Parsing current pile
        pile = state['current_round']['current_pile']
        round_initiator = pile['first_player']
        played_cards = pile['played_cards']
        for j in range(len(played_cards)):
            obs[36 + j, round_initiator] = 1
            obs[36 + j, 4 + self.reference_deck.index(card_from_str(played_cards[j]))] = 1

        # Parse current hand
        player = state['current_round']['player']
        hand = player['hand']
        player_id = player['id']
        obs[40, player_id] = 1
        for card in hand:
            obs[40, 4 + self.reference_deck.index(card_from_str(card))] = 1

        # Parsing scores and predictions
        obs[41, state['current_round']['current_player']] = 1
        for i, pred in enumerate(state['current_round']['predictions']):
            obs[41, i + 4] = pred if pred is not None else 0
        for i, score in enumerate(state['current_round']['round_scores']):
            obs[41, i + 8] = score

        # Encoding trump
        obs[41, 12 + self.suits.index(state['current_round']['trump'])] = 1

        return obs

    def get_payoffs(self) -> ndarray:
        """
        Returns the payoffs of the game, depending on the reward strategy configured.

        :return: A vector of shape [n_players] containing the score for each player.
        """
        return np.array(self.game.get_payoffs())

    def _decode_action(self, action_id: int) -> str:
        """
        Turns an action id (index on the card deck) into a string representation of the card. The string representation
        is required by the game.

        :param action_id: Index of the cared in the full deck
        :return: String representation of the card (i.e. S6 or DT).
        """
        card = self.reference_deck[action_id]
        return card.suit + card.rank

    def _get_legal_actions(self) -> dict:
        """
        Gets the legal actions in the current game state

        :return: A dictionary of all legal moves containing the action id as key.
        """
        legal_actions = self.game.get_legal_actions()
        legal_actions_ids = {self.reference_deck.index(card_from_str(a)): None for a in legal_actions}
        return OrderedDict(legal_actions_ids)

    def get_perfect_information(self) -> dict:
        """
        Extracts the full state of the game, not only the observation of the current player.

        :return: Dict containing the full state of the game.
        """
        return self.game.get_full_state()
