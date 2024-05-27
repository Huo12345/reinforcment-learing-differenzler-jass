import numpy as np
from itertools import combinations
from src.diff.utility import score_card, beats, get_full_deck, card_from_str

class GreedyAgent:
    def __init__(self, semi_greedy=False):
        self.semi_greedy = semi_greedy
        self.reference_deck = get_full_deck()
        self.use_raw = True

    def evaluate_state(self, state, action):
        """
        This function tries to estimate how many points are to be made from a pile when playing action
        """
        state = state['raw_obs']
        current_round = state['current_round']
        pile = [card_from_str(c) for c in current_round['current_pile']['played_cards']]
        trump = current_round['trump']
        
        # Check if action is beaten by any card in the pile
        if any(beats(action, card, trump) for card in pile):
            return 0  # No points scored if the action card is beaten

        hand = [card_from_str(c) for c in current_round['player']['hand']]
        played_piles = current_round['played_piles']
        
        # Get already played cards
        played_cards = [card_from_str(card) for pile in played_piles for card in pile['played_cards']]
        
        # Calculate the number of cards left to play in the current pile
        cards_left_to_play = 4 - len(pile) - 1
        
        # Find all cards not played yet (full deck - hand - pile - played_cards)
        full_deck = get_full_deck()
        for card in hand + pile + played_cards:
            full_deck.remove(card)
        not_played = full_deck
        
        # Find all cards not yet played that don't beat the action card
        scorable_cards = [card for card in not_played if not beats(action, card, trump)]
        
        # Calculate the potential score for the action card
        score = score_card(action, trump)
        pile_score = sum(score_card(card, trump) for card in pile)
        score += pile_score

        # Find all combinations of length cards_left_to_play in scorable_cards (pick without replacement)
        continuations = list(combinations(scorable_cards, cards_left_to_play))
        total_combinations = list(combinations(not_played, cards_left_to_play))

        # Calculate the expected score
        expected_score = 0
        for continuation in continuations:
            continuation_score = sum(score_card(card, trump) for card in continuation)
            prob = 1 / len(total_combinations)  # Assume uniform probability
            expected_score += prob * continuation_score

        return score + expected_score

    def select_action_probability(self, state):
        current_round = state['raw_obs']['current_round']
        legal_actions = [card_from_str(c) for c in current_round['legal_moves']]
        player = current_round['current_player']

        predicted_points = current_round['player']['prediction']
        scored_points = current_round['round_scores'][player]
        base_value = predicted_points - scored_points

        action_values = []
        for action in legal_actions:
            value = -abs(base_value - self.evaluate_state(state, action))
            action_values.append(value)

        if self.semi_greedy:
            # Sample from softmax probabilities
            probabilities = np.exp(action_values - np.max(action_values))  # for numerical stability
            probabilities /= np.sum(probabilities)
        else:
            # Greedy choice
            action_index = np.argmax(action_values)
            probabilities = [0 if i != action_index else 1 for i in range(len(legal_actions))]

        return probabilities

    def step(self, state):
        probabilities = self.select_action_probability(state)
        legal_actions = state['raw_obs']['current_round']['legal_moves']
        action_index = np.random.choice(len(legal_actions), p=probabilities)
        return legal_actions[action_index]

    def eval_step(self, state):
        return self.step(state), {}
