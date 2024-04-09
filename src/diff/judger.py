from functools import reduce

from rlcard.games.base import Card

from .player import DiffPlayer


class DiffJudger:

    def score_player(self, player: DiffPlayer) -> None:
        player.score = abs(player.round_score - player.prediction)
        player.round_score = 0
        player.prediction = 0

    def score_pile(self, cards: list[Card], trump: str, is_last: bool) -> int:
        pile_score = sum([self.score_card(c, trump) for c in cards])
        if is_last:
            pile_score += 5
        return pile_score

    def score_card(self, card: Card, trump: str) -> int:
        match card.rank:
            case 'A':
                return 11
            case 'K':
                return 4
            case 'Q':
                return 3
            case 'J':
                return 20 if card.suit == trump else 2
            case '10':
                return 10
            case '9':
                return 14 if card.suit == trump else 0
            case _:
                return 0

    def find_legal_moves(self, played: list[Card], hand: list[Card], trump: str) -> list[bool]:
        if len(played) == 0:
            return [True for _ in hand]

        active_suit = played[0].suit

        can_follow_suit = any([c.suit == active_suit for c in hand])
        is_trump_drawn = active_suit == trump
        highest_trump_played = self.highest_played_trump(played, trump)

        only_trump_is_jack = [c.rank for c in hand if c.suit == trump] == ['J']
        allowed_trumps = [c.suit == trump and self.beats(highest_trump_played, c, trump) for c in hand]

        if is_trump_drawn:
            return [True for _ in hand] if only_trump_is_jack else [c.suit == trump for c in hand]

        if can_follow_suit:
            same_suit = [c.suit == active_suit for c in hand]
            return list(map(lambda t: t[0] and t[1], zip(same_suit, allowed_trumps)))

        non_trump = [c.suit != trump for c in hand]
        return list(map(lambda t: t[0] and t[1], zip(non_trump, allowed_trumps)))

    def highest_played_trump(self, played: list[Card], trump: str) -> Card | None:
        trump_cards_played = [c for c in played if c.rank == trump]
        return reduce(lambda a, b: b if self.beats(a, b, trump) else a, trump_cards_played, None)

    def beats(self, first: Card, second: Card, trump: str) -> bool:
        """ Checks if the second card beats the first
        """
        if first is None or second is None:
            return first is None

        if first.suit != second.suit:
            return second.suit == trump

        if first.suit == trump:
            order = ['6', '7', '8', '10', 'Q', 'K', 'A', '9', 'J']
        else:
            order = ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

        return order.index(first.rank) < order.index(second.rank)
