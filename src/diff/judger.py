from .player import DiffPlayer


class DiffJudger:
    """
    Handles the distribution of points at the end of a round.
    """

    def score_player(self, player: DiffPlayer) -> None:
        """
        Calculates the difference between the prediction and the round score for the player and adds that to the
        overall score.

        :param player: Player to score
        """
        player.score += abs(player.round_score - player.prediction)
        player.round_score = 0
        player.prediction = 0
