import random
from copy import deepcopy

import torch
import numpy as np
from numpy import ndarray

from rlcard.utils import Card

from src.diff.utility import get_full_deck, find_legal_moves, takes_pile, score_card


class LookaheadDqnAgent:
    """
    Agent that uses a DQN and lookahead to decide which action to take. Only works on models trained on compressed
    state.
    """

    def __init__(self,
                 path: str,
                 device: torch.device = torch.device('cpu'),
                 look_ahead_depth: int = 1,
                 look_ahead_samples: int = 10,
                 discount_factor: float = 0.99):
        """
        Loads a pretrained DQN model.

        :param path: Path to the saved model.
        :param device: Torch device to use.
        """
        dqn_agent = torch.load(path, map_location=device)
        self.model = dqn_agent.q_estimator.qnet
        self.model.eval()
        self.device = device
        self.use_raw = False
        self.discount_factor = discount_factor
        self.look_ahead_depth = look_ahead_depth
        self.look_ahead_samples = look_ahead_samples
        self.reference_deck = get_full_deck()

    def step(self, state: dict) -> int:
        """
        Calculates the next step to be taken based on the partial observation for the current player.

        :param state: Observation of the current game state from the perspective of the current player.
        :return: The action to take.
        """
        q_values = self._predict(state)
        legal_actions = list(state['legal_actions'].keys())
        action_idx = legal_actions.index(np.argmax(q_values))

        return legal_actions[action_idx]

    def eval_step(self, state: dict) -> tuple[int, dict]:
        """
        Calculates the next step to be taken based on the partial observation for the current player.

        :param state: Observation of the current game state from the perspective of the current player.
        :return: Tuple containing the action to take and infos to analyse the decision.
        """
        q_values = self._predict(state)
        legal_actions = list(state['legal_actions'].keys())
        action_idx = legal_actions.index(np.argmax(q_values))

        info = {'values': {state['raw_legal_actions'][i]: float(q_values[list(state['legal_actions'].keys())[i]]) for i
                           in range(len(state['legal_actions']))}}

        return legal_actions[action_idx], info

    def _predict(self, state: dict) -> ndarray:
        """
        Calculates the masked q values based on the trained q network, legal actions and lookahead.

        :param state: Observation of the current game state from the perspective of the current player.
        :return: Masked q values, containing -infinity if the move is illegal, otherwise network prediction.
        """
        # Reading the observations and guessing a full state
        legal_actions = list(state['legal_actions'].keys())
        state = self._tensor_to_state(state['obs'])
        tree = self._sample_tree(state)

        # Make an estimate for each legal action based on look ahead
        masked_q_values = -np.inf * np.ones(len(self.reference_deck), dtype=float)
        for action in legal_actions:
            samples = [self._sample_state(state, action, tree) for _ in range(self.look_ahead_samples)]
            sample_predictions = [self._look_ahead(self.look_ahead_depth - 1, tree, state) for (state, tree) in samples]
            masked_q_values[action] = np.average(sample_predictions) * self.discount_factor

        return masked_q_values

    def _look_ahead(self, n: int, tree: list[list[Card]], state: dict) -> float:
        """
        Looks ahead n steps. If n == 0 then finds the best q value for the current state, otherwise samples
        look_ahead_samples of possible next states and averages the look ahead predictions for these.

        :param n: Number of steps to look ahead.
        :param tree: How the cards are distributed (list of players and their hands)
        :param state: Current state of the game
        :return: Estimate of the q value based on the look ahead action
        """
        # If there are no cards on hand the end of the round is reached and the true value of the state is known
        if len(state['hand']) == 0:
            return -1 * abs(state['scores'][state['id']] - state['predictions'][state['id']]) / 157.

        # Calculating the masked q values for the current state
        t = self._state_to_tensor(state)
        legal_actions = [self.reference_deck.index(c)
                         for c in find_legal_moves(state['pile'], state['hand'], state['trump'])]
        masked_q_values = self._calculate_masked_q_values(t, legal_actions)

        # Fining the best action based on the masked q values
        best_action = np.argmax(masked_q_values)

        # If is the last lookahead step, return the best q value
        if n == 0:
            return masked_q_values[best_action].item()

        # Sample look_ahead_samples possible continuations of the game
        samples = [self._sample_state(state, best_action, tree) for _ in range(self.look_ahead_samples)]

        # Average the look ahead values for the sampled continuations
        return np.average([self._look_ahead(n - 1, tree, state) for (state, tree) in samples]) * self.discount_factor

    def _sample_tree(self, state: dict) -> list[list[Card]]:
        """
        Creates an estimation of the full state of the game (which player holds which cards) to run a look ahead
        against.

        :param state: Current state of the game
        :return: Estimation for the distribution of the remaining cards
        """
        deck = get_full_deck()

        # Finding the cards not yet played and not held by the player
        deck = filter(lambda e: e not in state['played_cards'], deck)
        deck = filter(lambda e: e not in state['pile'], deck)
        deck = list(filter(lambda e: e not in state['hand'], deck))

        # Randomly order the remaining cards
        random.shuffle(deck)

        # Preparing variables for distributing the remaining cards
        cards_on_pile = len(state['pile'])
        cards_in_hand = len(state['hand'])
        hands = [[] for _ in range(4)]
        drawn_cards = 0

        # Distributing the cards to the different players
        for i in range(4):
            p = (state['id'] + i) % 4
            if p == state['id']:
                hands[p] = state['hand']
                continue
            cards_to_draw = cards_in_hand if i + cards_on_pile < 4 else cards_in_hand - 1
            hands[p] = deck[drawn_cards:drawn_cards + cards_to_draw]
            drawn_cards += cards_to_draw

        return hands

    def _sample_state(self, state: dict, action: int, tree: list[list[Card]]) -> tuple[dict, list[list[Card]]]:
        """
        Finds a possible continuation for the game when a specific action is taken.

        :param state: Current state of the game.
        :param action: Action taken.
        :param tree: How the cards are distributed (list of players and their hands).
        :return: The next observation of the game as well as the new distribution of the remaining cards.
        """
        # Copies the state
        state = deepcopy(state)
        tree = deepcopy(tree)

        # Finds the action to execute
        action_index = state['hand'].index(self.reference_deck[action])

        # Plays the remaining cards on the pile
        state['pile'].append(state['hand'].pop(action_index))
        tree[state['id']].pop(action_index)
        steps = 4 - len(state['pile'])
        self._play_radom_n_steps(state, steps, tree, state['id'] + 1)

        # Completes the pile
        pile_winner = (takes_pile(state['pile'], state['trump']) + state['id'] + steps + 1) % 4
        pile_score = sum([score_card(c, state['trump']) for c in state['pile']])
        state['played_cards'].extend(state['pile'])
        state['pile'] = []
        state['scores'][pile_winner] += pile_score

        # If this was the last pile to be played, adds the 5 point bonus to the last pile and returns the state
        if len(state['hand']) == 0:
            state['scores'][pile_winner] += 5
            return state, tree

        # Plays on the next pile until its the players turn again
        steps = (state['id'] - pile_winner) % 4
        self._play_radom_n_steps(state, steps, tree, pile_winner)

        return state, tree

    def _play_radom_n_steps(self, state: dict, steps: int, tree: list[list[Card]], start_player: int) -> None:
        """
        Plays the next step cards on the current pile randomly, modifies the state and tree accordingly.

        :param state: Current state of the game.
        :param steps: Number of steps to be played randomly.
        :param tree: How the cards are distributed (list of players and their hands).
        :param start_player: The player who starts playing the next steps.
        """
        for i in range(steps):
            player = (start_player + i) % 4
            legal_moves = find_legal_moves(state['pile'], tree[player], state['trump'])
            action = random.choice(legal_moves)
            action_index = tree[player].index(action)
            state['pile'].append(tree[player].pop(action_index))

    def _tensor_to_state(self, tensor: ndarray) -> dict:
        """
        Transforms an observation vector into an observation state dict.

        :param tensor: The observation vector.
        :return: Observation of the state as dict.
        """
        # Initializes utility variables
        deck = get_full_deck()
        suits = ['S', 'H', 'D', 'C']

        # Extracts string representations of played cards, hand and pile
        played = [deck[i] for i in range(len(tensor[0])) if tensor[0][i] == 1]
        pile = [deck[i] for i in range(len(tensor[1])) if tensor[1][i] == 1]
        hand = [deck[i] for i in range(len(tensor[2])) if tensor[2][i] == 1]

        # Extracts other game information
        player_id = np.argmax(tensor[3][0:4]).item()
        predictions = [e.item() for e in tensor[3][4:8]]
        scores = [e.item() for e in tensor[3][8:12]]
        trump = suits[np.argmax(tensor[3, 12:16])]

        # Puts extracted information into a dictionary
        return {
            'played_cards': played,
            'pile': pile,
            'hand': hand,
            'id': player_id,
            'predictions': predictions,
            'scores': scores,
            'trump': trump
        }

    def _state_to_tensor(self, state: dict) -> ndarray:
        """
        Transforms an observation state dict into an observation vector for estimating q values.

        :param state: Current state of the game.
        :return: Observation vector.
        """
        obs = np.zeros((4, 36), dtype=int)
        suits = ['S', 'H', 'D', 'C']

        # Parsing history
        for card in state['played_cards']:
            obs[0, self.reference_deck.index(card)] = 1

        # Parsing current pile
        for card in state['pile']:
            obs[1, self.reference_deck.index(card)] = 1

        # Parsing current hand
        for card in state['hand']:
            obs[2][self.reference_deck.index(card)] = 1

        # Parsing scores and predictions
        obs[3][state['id']] = 1
        for i, pred in enumerate(state['predictions']):
            obs[3][i + 4] = pred if pred is not None else 0
        for i, score in enumerate(state['scores']):
            obs[3][i + 8] = score

        # Encoding trump
        obs[3, 12 + suits.index(state['trump'])] = 1

        return np.expand_dims(obs, axis=0)

    def _calculate_masked_q_values(self, obs: ndarray, legal_actions: list[int]) -> ndarray:
        """
        Calculates the masked q values for a given observation vector.

        :param obs: Observation vector.
        :param legal_actions: Legal actions.
        :return: Q values with -infinity for illegal actions and the estimated q values for the legal actions.
        """
        with torch.no_grad():
            # Reshaping into single batch shape
            obs = np.expand_dims(obs, 0)

            # Calculating q values
            s = torch.from_numpy(obs).float().to(self.device)
            q = self.model(s).cpu().detach().numpy()[0]

            # Masking q values
            masked_q = -np.inf * np.ones(len(q), dtype=float)
            masked_q[legal_actions] = q[legal_actions]
            return masked_q
