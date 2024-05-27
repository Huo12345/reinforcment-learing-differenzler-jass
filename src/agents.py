import numpy as np
import random
from copy import deepcopy

from rlcard.agents import DQNAgent
from diff.utility import get_full_deck, find_legal_moves, takes_pile, score_card

class LookAheadDqnAgent(DQNAgent):

    def __init__(self,
                 replay_memory_size=20000,
                 replay_memory_init_size=100,
                 update_target_estimator_every=1000,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=20000,
                 batch_size=32,
                 num_actions=2,
                 state_shape=None,
                 train_every=1,
                 mlp_layers=None,
                 learning_rate=0.00005,
                 device=None,
                 save_path=None,
                 save_every=float('inf'),
                 look_ahead_depth=1,
                 look_ahead_samples=100,
                 look_ahead_train=True,
                 look_ahead_eval=False):
        super().__init__(replay_memory_size, replay_memory_init_size, update_target_estimator_every, discount_factor,
                         epsilon_start, epsilon_end, epsilon_decay_steps, batch_size, num_actions, state_shape,
                         train_every, mlp_layers, learning_rate, device, save_path, save_every)
        self.look_ahead_depth = look_ahead_depth
        self.look_ahead_samples = look_ahead_samples
        self.look_ahead_tree = 1
        self.look_ahead_train = look_ahead_train
        self.look_ahead_eval = look_ahead_eval
        self.reference_deck = get_full_deck()

    def train(self):
        if not self.look_ahead_train:
            super(LookAheadDqnAgent, self).train()

        state_batch, action_batch, reward_batch, next_state_batch, done_batch, legal_actions_batch = self.memory.sample()

        target_batch = np.array([self._look_ahead_sampling(s) for s in next_state_batch])

        # Perform gradient descent update
        state_batch = np.array(state_batch)

        loss = self.q_estimator.update(state_batch, action_batch, target_batch)
        print('\rINFO - Step {}, rl-loss: {}'.format(self.total_t, loss), end='')

        # Update the target estimator
        if self.train_t % self.update_target_estimator_every == 0:
            self.target_estimator = deepcopy(self.q_estimator)
            print("\nINFO - Copied model parameters to target network.")

        self.train_t += 1

        if self.save_path and self.train_t % self.save_every == 0:
            # To preserve every checkpoint separately,
            # add another argument to the function call parameterized by self.train_t
            self.save_checkpoint(self.save_path)
            print("\nINFO - Saved model checkpoint.")

    def predict(self, state):
        if not self.look_ahead_eval:
            return super(LookAheadDqnAgent, self).predict(state)

        legal_actions = list(state['legal_actions'].keys())
        state = self._tensor_to_state(state['obs'])
        tree = self._sample_tree(state)

        masked_q_values = -np.inf * np.ones(self.num_actions, dtype=float)
        for action in legal_actions:
            samples = [self._sample_state(state, action, tree) for _ in range(self.look_ahead_samples)]
            sample_predictions = [self._look_ahead(self.look_ahead_depth - 1, tree, state) for (state, tree) in samples]
            masked_q_values[action] = np.average(sample_predictions) * self.discount_factor

        return masked_q_values

    def _look_ahead_sampling(self, state):
        state = self._tensor_to_state(state)

        tree = self._sample_tree(state)
        return self._look_ahead(self.look_ahead_depth, tree, state)

    def _look_ahead(self, n, tree, state):
        if len(state['hand']) == 0:
            return 1 * abs(state['scores'][state['id']] - state['predictions'][state['id']]) / 157.

        t = self._state_to_tensor(state)
        q_values = self.q_estimator.predict_nograd(t)[0]

        masked_q_values = -np.inf * np.ones(self.num_actions, dtype=float)
        legal_actions = [self.reference_deck.index(c) for c in find_legal_moves(state['pile'], state['hand'], state['trump'])]
        masked_q_values[legal_actions] = q_values[legal_actions]
        best_action = np.argmax(masked_q_values)

        if n == 0:
            q_values = self.target_estimator.predict_nograd(t)[0]
            return q_values[best_action]

        samples = [self._sample_state(state, best_action, tree) for _ in range(self.look_ahead_samples)]

        return np.average([self._look_ahead(n - 1, tree, state) for (state, tree) in samples]) * self.discount_factor

    def _sample_tree(self, state):
        deck = get_full_deck()
        deck = filter(lambda e: e not in state['played_cards'], deck)
        deck = filter(lambda e: e not in state['pile'], deck)
        deck = list(filter(lambda e: e not in state['hand'], deck))
        random.shuffle(deck)

        cards_on_pile = len(state['pile'])
        cards_in_hand = len(state['hand'])
        hands = [[] for _ in range(4)]
        drawn_cards = 0

        for i in range(4):
            p = (state['id'] + i) % 4
            if p == state['id']:
                hands[p] = state['hand']
                continue
            cards_to_draw = cards_in_hand if i + cards_on_pile < 4 else cards_in_hand - 1
            hands[p] = deck[drawn_cards:drawn_cards + cards_to_draw]
            drawn_cards += cards_to_draw

        return hands

    def _sample_state(self, state, action, tree):
        state = deepcopy(state)
        tree = deepcopy(tree)
        action_index = state['hand'].index(self.reference_deck[action])

        state['pile'].append(state['hand'].pop(action_index))
        tree[state['id']].pop(action_index)
        steps = 4 - len(state['pile'])
        self._play_radom_n_steps(state, steps, tree, state['id'] + 1)

        pile_winner = (takes_pile(state['pile'], state['trump']) + state['id'] + steps + 1) % 4
        pile_score = sum([score_card(c, state['trump']) for c in state['pile']])
        state['played_cards'].extend(state['pile'])
        state['pile'] = []
        state['scores'][pile_winner] += pile_score

        if len(state['hand']) == 0:
            return state, tree

        steps = (state['id'] - pile_winner) % 4
        self._play_radom_n_steps(state, steps, tree, pile_winner)

        return state, tree

    def _play_radom_n_steps(self, state, steps, tree, start_player):
        for i in range(steps):
            player = (start_player + i) % 4
            legal_moves = find_legal_moves(state['pile'], tree[player], state['trump'])
            action = random.choice(legal_moves)
            action_index = tree[player].index(action)
            state['pile'].append(tree[player].pop(action_index))

    def _tensor_to_state(self, tensor):
        deck = get_full_deck()
        suits = ['S', 'H', 'D', 'C']

        played = [deck[i] for i in range(len(tensor[0])) if tensor[0][i] == 1]
        pile = [deck[i] for i in range(len(tensor[1])) if tensor[1][i] == 1]
        hand = [deck[i] for i in range(len(tensor[2])) if tensor[2][i] == 1]
        player_id = np.argmax(tensor[3][0:4]).item()
        predictions = [e.item() for e in tensor[3][4:8]]
        scores = [e.item() for e in tensor[3][8:12]]
        trump = suits[np.argmax(tensor[3, 12:16])]

        return {
            'played_cards': played,
            'pile': pile,
            'hand': hand,
            'id': player_id,
            'predictions': predictions,
            'scores': scores,
            'trump': trump
        }

    def _state_to_tensor(self, state):
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
