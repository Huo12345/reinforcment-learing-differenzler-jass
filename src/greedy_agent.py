import numpy as np
from itertools import combinations
from src.diff.utility import score_card, beats, get_full_deck

class GreedyAgent:
    def __init__(self, semi_greedy=False):
        self.semi_greedy = semi_greedy

    def evaluate_state(self, state, action):
        """
        This function tries to estimate how many points are to be made from a pile when playing action
        """
        current_round = state['current_round']
        pile = current_round['current_pile']['played_cards']
        trump = current_round['trump']
        player_id = state['player']['id']  # Get player_id from state
        
        # Check if action is beaten by any card in the pile
        if any(beats(action, card, trump) for card in pile):
            return 0  # No points scored if the action card is beaten

        hand = current_round['player']['hand']
        played_piles = current_round['played_piles']
        
        # Get already played cards
        played_cards = [card for pile in played_piles for card in pile['played_cards']]
        
        # Calculate the number of cards left to play in the current pile
        num_players = state['num_players']
        cards_left_to_play = num_players - len(pile) - 1
        
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

    def select_action(self, state):
        current_round = state['current_round']
        legal_actions = current_round['legal_moves']

        predicted_points = current_round['player']['prediction']
        scored_points = current_round['player']['round_score']
        base_value = predicted_points - scored_points

        action_values = []
        for action in legal_actions:
            value = -abs(base_value - self.evaluate_state(state, action))
            action_values.append(value)

        if self.semi_greedy:
            # Sample from softmax probabilities
            probabilities = np.exp(action_values - np.max(action_values))  # for numerical stability
            probabilities /= np.sum(probabilities)
            action_index = np.random.choice(len(legal_actions), p=probabilities)
        else:
            # Greedy choice
            action_index = np.argmax(action_values)

        return legal_actions[action_index]

    def step(self, state):
        return self.select_action(state)