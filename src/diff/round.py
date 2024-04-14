from rlcard.games.base import Card

from .dealer import DiffDealer
from .pile import DiffPile
from .player import DiffPlayer


class DiffRound:
    def __init__(self, n_piles: int, n_players: int, first_player: int, dealer: DiffDealer) -> None:
        self.n_piles = n_piles
        self.pile = 0
        self.current_pile: DiffPile | None = None
        self.played_piles = []
        self.n_players = n_players
        self.first_player = first_player
        self.current_player = first_player
        self.dealer = dealer
        self.trump: str | None = None
        self.trump_card: Card | None = None

    def deal_cards(self, players: list[DiffPlayer]) -> bool:
        if self.trump is not None:
            return False

        self.dealer.shuffle()
        cards_per_player = len(self.dealer.deck) // self.n_players
        for i in range(self.n_players):
            self.dealer.deal_cards(players[self.current_player + i % self.n_players], cards_per_player)
        self.trump_card = self.dealer.show_last_card()
        self.trump = self.trump_card.suit
        self.current_pile = DiffPile(self.first_player, self.n_players, self.trump)
        return True

    def proceed_round(self, players: list[DiffPlayer], action: int) -> bool:
        if self.is_over():
            return False

        card = players[self.current_player].hand.pop(action)
        self.current_pile.play(card)
        self.current_player = self.current_pile.get_current_player()
        if not self.current_pile.is_done():
            return True

        pile_winner = self.current_pile.takes_turn()
        players[self.current_player].round_score += self.current_pile.get_score()
        self.played_piles.append(self.current_pile)
        self.current_player = pile_winner
        self.pile += 1
        self.current_pile = None if self.is_over() else DiffPile(self.current_player, self.n_players, self.trump)
        return True

    def is_over(self) -> bool:
        return self.pile >= self.n_piles

    def get_state(self, players: list[DiffPlayer]) -> dict:
        return {
            "current_pile": self.current_pile.get_state(),
            "current_player": self.current_player,
            "played_piles": [p.get_state() for p in self.played_piles],
            "player_hand": players[self.current_player].hand,
        }

