import numpy as np

from rlcard.games.base import Card

from src.diff.player import DiffPlayer


class DiffDealer:

    def __init__(self, np_random: np.random) -> None:
        self.np_random = np_random
        self.deck = self.init_deck()
        self.shuffle()
        self.deal_ptr = 0

    def init_deck(self) -> list[Card]:
        suits = ['S', 'H', 'D', 'C']
        ranks = ['6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        return [Card(s, r) for s in suits for r in ranks]

    def shuffle(self) -> None:
        shuffle_deck = np.array(self.deck)
        self.np_random.shuffle(shuffle_deck)
        self.deck = list(shuffle_deck)

    def deal_cards(self, player: DiffPlayer) -> None:
        player.hand.append(self.deck[self.deal_ptr:self.deal_ptr + 3])
        self.deal_ptr += 3

    def show_last_card(self) -> Card:
        return self.deck[-1]
