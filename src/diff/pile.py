from rlcard.games.base import Card

from .utility import score_card, takes_pile


class DiffPile:
    """
    Represents a single pile in a round of differenzler. Each player has to play a single card and at the end the player
    with the strongest card played takes the piles and its points.
    """

    def __init__(self, first_player: int, number_of_players: int, trump: str, is_last: bool = False) -> None:
        """
        Initializes a new pile

        :param first_player: First player to play on the pile. In the first round it's the first player of the round and
            after that it's the player that won the last pile.
        :param number_of_players: Number of players in the game
        :param trump: Trump suit for the current game
        :param is_last: Indicates if it is the last pile. If so it adds 5 points to the piles score.
        """
        self.first_player = first_player
        self.current_player = first_player
        self.number_of_players = number_of_players
        self.trump = trump
        self.pile: list[Card] = []
        self.score = 5 if is_last else 0

    def play(self, card: Card) -> None:
        """
        Plays a card on the pile and adds its points to the score.

        :param card: Card played by the current player
        """
        if self.is_done():
            raise Exception("Cannot play on a completed pile")

        self.pile.append(card)
        self.current_player = (self.current_player + 1) % self.number_of_players
        self.score += score_card(card, self.trump)

    def is_done(self) -> bool:
        """
        Indicates if the pile is done.

        :return: True if all player have played for this pile
        """
        return len(self.pile) == self.number_of_players

    def takes_turn(self) -> int:
        """
        Returns the player who takes the pile.

        :return: Id of the player who takes the pile
        """
        if not self.is_done():
            raise Exception("Cannot take pile if the pile is not done")
        return (takes_pile(self.pile, self.trump) + self.current_player) % self.number_of_players

    def get_score(self) -> int:
        """
        Returns the score of the pile.

        :return: Score of the pile
        """
        return self.score

    def get_current_player(self) -> int:
        """
        Returns the player who plays next

        :return: Id of the player who plays next
        """
        return self.current_player

    def get_pile(self) -> list[Card]:
        """
        Returns the cards on the pile.

        :return: Cards on the pile
        """
        return self.pile

    def get_state(self) -> dict:
        """
        Transforms the state of the pile into a dictionary.

        :return: State of the pile as a dict
        """
        return {
            "first_player": self.first_player,
            "played_cards": [c.get_index() for c in self.pile],
            "score": self.score
        }
