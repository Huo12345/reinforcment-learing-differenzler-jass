from functools import reduce

from rlcard.games.base import Card


def score_card(card: Card, trump: str) -> int:
    """
    Calculates the score of a given card.

    :param card: Card to be scored.
    :param trump: Trump suit in the current round.
    :return: The points this card is worth.
    """
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
    """
    Calculates which card of the pile is the strongest and takes the pile.

    :param pile: All cards on the pile.
    :param trump: Trump suit in the current round.
    :return: Index of the strongest card.
    """
    takes = 0
    for i in range(1, len(pile)):
        takes = i if beats(pile[takes], pile[i], trump) else takes
    return takes


def beats(first: Card, second: Card, trump: str) -> bool:
    """
    Checks if the second card beats the first.

    :param first: First card.
    :param second: Second card.
    :param trump: Trump suit in the current round.
    :return: True if the second card beats the first card, False otherwise.
    """
    # Checking edge cases
    if first is None or second is None:
        return first is None

    # Checking if the suits don't match. Then the second only takes if it is trump.
    if first.suit != second.suit:
        return second.suit == trump

    # Checks which card is stronger within the same suit. Different order for trump suit.
    if first.suit == trump:
        order = ['6', '7', '8', 'T', 'Q', 'K', 'A', '9', 'J']
    else:
        order = ['6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

    return order.index(first.rank) < order.index(second.rank)


def find_legal_moves(pile: list[Card], hand: list[Card], trump: str) -> list[Card]:
    """
    Finds all legal moves for a player.

    :param pile: All cards on the pile.
    :param hand: The hand of the current player.
    :param trump: Trump suit in the current round.
    :return: List of all cards that can be played.
    """
    # Copying the hand to not affect the game state.
    hand = [c for c in hand]

    # If the player is the first to play on the given pile, all cards on the hand can be played.
    if len(pile) == 0:
        return hand

    # Determining properties of the current pile and hand
    active_suit = pile[0].suit

    can_follow_suit = any([c.suit == active_suit for c in hand])
    is_trump_drawn = active_suit == trump
    highest_trump_played = highest_played_trump(pile, trump)

    only_trump_is_jack = [c.rank for c in hand if c.suit == trump] == ['J']
    allowed_trumps = [c for c in hand if c.suit == trump and beats(highest_trump_played, c, trump)]

    # If trump is the suit out, then the player must play any trump card if they have one on hand. Exception is the
    # jack, which can never be forced. So if the players only trump card is a jack, then they can play whatever.
    if is_trump_drawn and can_follow_suit:
        return hand if only_trump_is_jack else [c for c in hand if c.suit == trump]

    # If the suit out isn't trump and the player can follow suit, they can eiter play any card of the suit or play a
    # stronger trump card than what's already played.
    if can_follow_suit:
        same_suit = [c for c in hand if c.suit == active_suit]
        return list(set(same_suit) | set(allowed_trumps))

    # If the player can't follow suit, all cards except trump cards lower than the strongest trump card are allowed. If
    # this rules out all cards in hand, then all cards can be played.
    non_trump = [c for c in hand if c.suit != trump]
    legal_moves = list(set(non_trump) | set(allowed_trumps))
    return legal_moves if len(legal_moves) != 0 else hand


def highest_played_trump(played: list[Card], trump: str) -> Card | None:
    """
    Finds the highest played trump card on the pile.

    :param played: All cards on the pile.
    :param trump: Trump suit in the current round.
    :return: None if no trump card was played, otherwise the strongest trump card on the pile.
    """
    trump_cards_played = [c for c in played if c.suit == trump]
    return reduce(lambda a, b: b if beats(a, b, trump) else a, trump_cards_played, None)


def get_full_deck() -> list[Card]:
    """
    Creates a full deck of cards (6 to A for each suit, giving 36 cards in total).

    :return: A full deck of cards.
    """
    suits = ['S', 'H', 'D', 'C']
    ranks = ['6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    return [Card(s, r) for s in suits for r in ranks]


def card_from_str(card: str) -> Card:
    """
    Creates a card instance from its string representation.

    :param card: String representation of a card.
    :return: Card instance.
    """
    return Card(card[0], card[1])


def followed_suit(piles: list[dict], n_players: int, trump: str) -> list[list[int]]:
    """
    Calculates based on the history of played piles which player has followed suit for which suit. Following suit means
    has either played a card which suit matches the first card on the pile or is a trump card.

    :param piles: All piles played so far.
    :param n_players: Number of players in the game.
    :param trump: Trump suit in the current round.
    :return: A list of shape [4, n_players] containing a 1 on a players position if the player hasn't followed suit in
    the past, 0 otherwise. The suit order is 'S', 'H', 'D', 'C'.
    """
    suits = ['S', 'H', 'D', 'C']
    result = [[0 for _ in range(n_players)] for _ in suits]

    for pile in piles:
        suit = card_from_str(pile['played_cards'][0]).suit
        for i, card in enumerate([card_from_str(c) for c in pile['played_cards']]):
            if card.suit != suit and card.suit != trump:
                result[suits.index(suit)][(pile['first_player'] + i) % n_players] = 1

    return result


def find_strong_cards(piles: list[dict], hand: list[str], trump: str) -> list[str]:
    """
    Finds all cards on hand that can only be taken by a trump card.

    :param piles: All piles played so far.
    :param hand: All cards on hand.
    :param trump: Trump suit of the current round.
    :return: A list of cards on hand that can only be taken by a trump card.
    """
    played_cards = [card_from_str(card) for pile in piles for card in pile['played_cards']]

    hidden_cards = [card for card in get_full_deck() if card.get_index() not in hand and card not in played_cards]

    good_cards = [card for card in hand if not any(
        [beats(card_from_str(card), other, trump) for other in hidden_cards if card_from_str(card).suit == other.suit])]

    return good_cards


def find_weak_cards(piles: list[dict], hand: list[str], trump: str) -> list[str]:
    """
    Finds all cards on hand that have to be taken by another player when played.

    :param piles: All piles played so far.
    :param hand: All cards on hand.
    :param trump: Trump suit of the current round.
    :return: A list of cards on hand another player has to take when played.
    """
    played_cards = [card_from_str(card) for pile in piles for card in pile['played_cards']]

    hidden_cards = [card for card in get_full_deck() if card.get_index() not in hand and card not in played_cards]

    weak_cards = [card for card in hand if all(
        [beats(card_from_str(card), other, trump) for other in hidden_cards if card_from_str(card).suit == other.suit])]

    return weak_cards
