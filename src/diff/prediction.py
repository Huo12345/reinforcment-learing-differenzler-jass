
class PredictionStrategy:
    def get_prediction(self, player: int, game_state: dict) -> int:
        pass


class CompositePredictionStrategy(PredictionStrategy):

    def __int__(self, player_strategies: list[PredictionStrategy]):
        self.player_strategies = player_strategies

    def get_prediction(self, player: int, game_state: dict) -> int:
        return self.player_strategies[player].get_prediction(player, game_state)
