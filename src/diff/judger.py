from functools import reduce

from rlcard.games.base import Card

from .player import DiffPlayer


class DiffJudger:

    def score_player(self, player: DiffPlayer) -> None:
        player.score += abs(player.round_score - player.prediction)
        player.round_score = 0
        player.prediction = 0
