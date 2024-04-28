from collections import OrderedDict

import numpy as np
from numpy import ndarray
from rlcard.envs import Env

from .game import DiffGame
from .prediction import FixedPredictionStrategy
from .utility import get_full_deck

DEFAULT_GAME_CONFIG = {
    'players': 4,
    'rounds': 2,
    'prediction_strategy': FixedPredictionStrategy(157 // 4)
}


class DivEnv(Env):

    def __init__(self, config) -> None:
        self.name = 'differenzler'
        self.default_game_config = DEFAULT_GAME_CONFIG
        self.game = DiffGame()
        super().__init__(config)
        self.state_shape = [[41, 40] for _ in range(self.num_players)]
        self.action_shape = [36 for _ in range(self.num_players)]
        self.reference_deck = get_full_deck()

    def _extract_state(self, state: dict) -> dict:
        return {
            'obs': self._parse_state_to_ndarray(state),
            'legal_actions': self._get_legal_actions(),
            'raw_obs': state,
            'raw_legal_actions': state['current_round']['legal_moves'],
            'action_record': self.action_recorder
        }

    def _parse_state_to_ndarray(self, state: dict) -> ndarray:
        obs = np.zeros((41, 40), dtype=int)

        # Paring history
        for i in range(len(state['current_round']['played_piles'])):
            pile = state['current_round']['played_piles'][i]
            round_initiator = pile['first_player']
            played_cards = pile['played_cards']
            for j in range(len(played_cards)):
                obs[i * 4 + j][round_initiator] = 1
                obs[i * 4 + j][4 + self.reference_deck.index(played_cards[j])] = 1

        # Parsing current pile
        pile = state['current_round']['current_pile']
        round_initiator = pile['first_player']
        played_cards = pile['played_cards']
        for j in range(len(played_cards)):
            obs[36 + j][round_initiator] = 1
            obs[36 + j][4 + self.reference_deck.index(played_cards[j])] = 1

        # Parse current hand
        player = state['current_round']['player']
        hand = player['hand']
        player_id = player['id']
        obs[40][player_id] = 1
        for card in hand:
            obs[40][4 + self.reference_deck.index(card)] = 1
        return obs

    def get_payoffs(self) -> ndarray:
        return np.array(self.game.get_payoffs())

    def _decode_action(self, action_id: int) -> str:
        card = self.reference_deck[action_id]
        return card.suit + card.rank

    def _get_legal_actions(self) -> dict:
        legal_actions = self.game.get_legal_actions()
        legal_actions_ids = {self.reference_deck.index(a): None for a in legal_actions}
        return OrderedDict(legal_actions_ids)

    def get_perfect_information(self) -> dict:
        return self.game.get_full_state()
