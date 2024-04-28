from rlcard.games.base import Card

from .dealer import DiffDealer
from .pile import DiffPile
from .player import DiffPlayer
from .utility import find_legal_moves


class DiffRound:
    def __init__(self, n_players: int, first_player: int, dealer: DiffDealer) -> None:
        self.n_piles = len(dealer.deck) / n_players
        self.pile = 0
        self.current_pile: DiffPile | None = None
        self.played_piles = []
        self.n_players = n_players
        self.first_player = first_player
        self.current_player = first_player
        self.dealer = dealer
        self.trump: str | None = None
        self.trump_card: Card | None = None
        self.predictions = 0

    def deal_cards(self, players: list[DiffPlayer]) -> None:
        if self.trump is not None:
            raise Exception("Cannot deal cards if they were already delt")

        self.dealer.shuffle()
        cards_per_player = len(self.dealer.deck) // self.n_players
        for i in range(self.n_players):
            self.dealer.deal_cards(players[(self.current_player + i) % self.n_players], cards_per_player)
            players[(self.current_player + i) % self.n_players].round_score = 0
        self.trump_card = self.dealer.show_last_card()
        self.trump = self.trump_card.suit
        self.pile += 1
        self.current_pile = DiffPile(self.first_player, self.n_players, self.trump)

    def make_prediction(self, players: list[DiffPlayer], prediction: int) -> None:
        if self.predictions_over():
            raise Exception("Cannot make a prediction if the prediction round is already over")
        players[(self.first_player + self.predictions) % self.n_players].prediction = prediction
        self.predictions += 1

    def proceed_round(self, players: list[DiffPlayer], action: int) -> None:
        if self.is_over() or not self.predictions_over():
            raise Exception("Cannot play if the predictions weren't made"
                            if self.predictions_over() else "Cannot play if the round is already over")

        card = players[self.current_player].hand.pop(action)
        self.current_pile.play(card)
        self.current_player = self.current_pile.get_current_player()
        if not self.current_pile.is_done():
            return

        pile_winner = self.current_pile.takes_turn()
        players[pile_winner].round_score += self.current_pile.get_score()
        self.played_piles.append(self.current_pile)
        self.current_player = pile_winner
        self.pile += 1
        if not self.is_over():
            self.current_pile = DiffPile(self.current_player, self.n_players, self.trump, self.pile == self.n_piles)

    def is_over(self) -> bool:
        return self.pile > self.n_piles

    def predictions_over(self):
        return self.predictions >= self.n_players

    def get_state(self, players: list[DiffPlayer], player: int) -> dict:
        predictions = [p.prediction for p in players]
        if not self.predictions_over():
            for i in range(self.n_players - self.predictions):
                predictions[self.first_player - i - 1 % self.n_players] = None

        return {
            "trump": self.trump,
            "trump_card": self.trump_card,
            "first_player": self.first_player,
            "current_pile": self.current_pile.get_state(),
            "current_player": self.current_player,
            "played_piles": [p.get_state() for p in self.played_piles],
            "player": players[player].get_state(),
            "predictions": predictions,
            "round_scores": [p.round_score for p in players],
            "legal_moves": find_legal_moves(self.current_pile.pile, players[player].hand, self.trump)
        }

    def get_legal_actions(self, players: list[DiffPlayer], player: int) -> list[Card]:
        return find_legal_moves(self.current_pile.pile, players[player].hand, self.trump)

    def get_full_state(self, players: list[DiffPlayer]):
        state = self.get_state(players, self.current_player)
        del state['player']
        state['players'] = [p.get_state() for p in players]
