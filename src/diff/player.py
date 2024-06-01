class DiffPlayer:
    """
    Holds the state of a player.
    """

    def __init__(self, player_id: int) -> None:
        """
        Initializes a player.

        :param player_id: Id of the player
        """
        self.player_id = player_id
        self.hand = []
        self.score = 0
        self.prediction = 0
        self.round_score = 0

    def get_player_id(self):
        """
        Returns the player's id.

        :return: Id of the player
        """
        return self.player_id

    def get_state(self):
        """
        Transforms the state of the player into a dictionary.

        :return: State of the player as a dict
        """
        return {
            "id": self.player_id,
            "hand": [c.get_index() for c in self.hand],
            "score": self.score,
            "prediction": self.prediction,
        }
