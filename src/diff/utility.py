from functools import reduce

from rlcard.games.base import Card


def score_card(card: Card, trump: str) -> int:
    match card.rank:
        case 'A':
            return 11
        case 'K':
            return 4
        case 'Q':
            return 3
        case 'J':
            return 20 if card.suit == trump else 2
        case 'T':
            return 10
        case '9':
            return 14 if card.suit == trump else 0
        case _:
            return 0


def takes_pile(pile: list[Card], trump: str) -> int:
    takes = 0
    for i in range(1, len(pile)):
        takes = i if beats(pile[takes], pile[i], trump) else takes
    return takes


def beats(first: Card, second: Card, trump: str) -> bool:
    """ Checks if the second card beats the first
    """
    if first is None or second is None:
        return first is None

    if first.suit != second.suit:
        return second.suit == trump

    if first.suit == trump:
        order = ['6', '7', '8', 'T', 'Q', 'K', 'A', '9', 'J']
    else:
        order = ['6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

    return order.index(first.rank) < order.index(second.rank)


def find_legal_moves(pile: list[Card], hand: list[Card], trump: str) -> list[Card]:
    if len(pile) == 0:
        return hand

    active_suit = pile[0].suit

    can_follow_suit = any([c.suit == active_suit for c in hand])
    is_trump_drawn = active_suit == trump
    highest_trump_played = highest_played_trump(pile, trump)

    only_trump_is_jack = [c.rank for c in hand if c.suit == trump] == ['J']
    allowed_trumps = [c for c in hand if c.suit == trump and beats(highest_trump_played, c, trump)]

    if is_trump_drawn and can_follow_suit:
        return hand if only_trump_is_jack else [c for c in hand if c.suit == trump]

    if can_follow_suit:
        same_suit = [c for c in hand if c.suit == active_suit]
        return list(set(same_suit) | set(allowed_trumps))

    non_trump = [c for c in hand if c.suit != trump]
    legal_moves = list(set(non_trump) | set(allowed_trumps))
    return legal_moves if len(legal_moves) != 0 else hand


def highest_played_trump(played: list[Card], trump: str) -> Card | None:
    trump_cards_played = [c for c in played if c.suit == trump]
    return reduce(lambda a, b: b if beats(a, b, trump) else a, trump_cards_played, None)
