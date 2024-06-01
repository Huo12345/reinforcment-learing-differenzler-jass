import numpy as np

from rlcard.games.base import Card

from .player import DiffPlayer
from .utility import get_full_deck


class DiffDealer:
    """
    Prepares the deck for 3, 4 or 5 players, shuffles the cards and distributes the cards to the players. Shows the last
    card delt to every player ot announce the trump suit.
    """

    def __init__(self, np_random: np.random, n_players: int) -> None:
        """
        Initializes the dealer

        :param np_random: Random state shared throughout the game, for reproducibility
        :param n_players: Number of players for this game
        """
        if n_players not in [3, 4, 5]:
            raise Exception('Allowed numer of players are 3, 4 or 5. {} players are not allowed'.format(n_players))
        self.np_random = np_random
        self.deck = self._init_deck(n_players == 5)
        self.shuffle()
        self.reference_dek = self._init_deck(n_players == 5)
        self.deal_ptr = 0

    @staticmethod
    def _init_deck(five_players: bool) -> list[Card]:
        """
        Creates a deck of cares according to the number of players. Removes the first card from the deck (S6) if there
        are 5 players.

        :param five_players: If true initializes a deck with 35 rather than 36 cards, used for the 5 player mode.
        :return: A list of Cards representing the deck to be played with.
        """
        deck = get_full_deck()
        if five_players:
            deck = deck[1:]
        return deck

    def shuffle(self) -> None:
        """
        Shuffles the deck.
        """
        shuffle_deck = np.array(self.deck)
        self.np_random.shuffle(shuffle_deck)
        self.deck = list(shuffle_deck)
        self.deal_ptr = 0

    def deal_cards(self, player: DiffPlayer, n_cards: int) -> None:
        """
        Deals n_cards cards to the player.

        :param player: The player to receive the cards.
        :param n_cards: Number of cards to be delt to the player.
        """
        player_cards = self.deck[self.deal_ptr:self.deal_ptr + n_cards]
        sorted_player_cards = sorted(player_cards, key=lambda c: self.reference_dek.index(c))
        player.hand.extend(sorted_player_cards)
        self.deal_ptr += n_cards

    def show_last_card(self) -> Card:
        """
        Shows the last card in the deck to announce the trump suit.
        """
        return self.deck[-1]
