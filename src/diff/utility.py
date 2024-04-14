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
        case '10':
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
        order = ['6', '7', '8', '10', 'Q', 'K', 'A', '9', 'J']
    else:
        order = ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

    return order.index(first.rank) < order.index(second.rank)
