import numpy as np

from rlcard.games.base import Card

from src.diff.player import DiffPlayer


class DiffDealer:

    def __init__(self, np_random: np.random, n_players: int) -> None:
        if n_players not in [3, 4, 5]:
            raise Exception('Allowed numer of players are 3, 4 or 5. {} players are not allowed'.format(n_players))
        self.np_random = np_random
        self.deck = self.init_deck(n_players == 5)
        self.shuffle()
        self.reference_dek = self.init_deck(n_players == 5)
        self.deal_ptr = 0

    def init_deck(self, five_players: bool) -> list[Card]:
        suits = ['S', 'H', 'D', 'C']
        ranks = ['6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        deck = [Card(s, r) for s in suits for r in ranks]
        if five_players:
            deck = deck[1:]
        return deck

    def shuffle(self) -> None:
        shuffle_deck = np.array(self.deck)
        self.np_random.shuffle(shuffle_deck)
        self.deck = list(shuffle_deck)
        self.deal_ptr = 0

    def deal_cards(self, player: DiffPlayer, n_cards: int) -> None:
        player.hand.extend(sorted(self.deck[self.deal_ptr:self.deal_ptr + n_cards], key=lambda c: self.reference_dek.index(c)))
        self.deal_ptr += n_cards

    def show_last_card(self) -> Card:
        return self.deck[-1]
