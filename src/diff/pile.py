from rlcard.games.base import Card

from .utility import score_card, takes_pile


class DiffPile:

    def __init__(self, first_player: int, number_of_players: int, trump: str, is_last: bool = False) -> None:
        self.first_player = first_player
        self.current_player = first_player
        self.number_of_players = number_of_players
        self.trump = trump
        self.pile: list[Card] = []
        self.score = 5 if is_last else 0

    def play(self, card: Card) -> bool:
        if not self.is_done():
            return False

        self.pile.append(card)
        self.current_player = self.current_player + 1 % self.number_of_players
        self.score += score_card(card, self.trump)
        return True

    def is_done(self) -> bool:
        return len(self.pile) == self.number_of_players

    def takes_turn(self) -> int | None:
        if not self.is_done():
            return None
        return takes_pile(self.pile, self.trump) + self.current_player % self.number_of_players

    def get_score(self) -> int:
        return self.score

    def get_current_player(self) -> int:
        return self.current_player

    def get_pile(self) -> list[Card]:
        return self.pile

    def get_state(self) -> dict:
        return {
            "first_player": self.first_player,
            "played_cards": self.pile,
            "score": self.score
        }
