import random

from diff import DiffGame, PredictionStrategy

from rlcard.utils.utils import print_card


class FixedPredictinStrategy(PredictionStrategy):
    def get_prediction(self, player: int, game_state: dict) -> int:
        return 157 // 4


def display_state(state: dict) -> None:
    print()
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print()

    print("Scores: %s" % (state['player_scores']))
    print("Predictions: %s" % (state['current_round']['predictions']))
    print("Round scores: %s" % (state['current_round']['round_scores']))
    print("Player %d turn" % state['current_round']['current_player'])
    print("===================== Pile =====================")
    print_card(state['current_round']['current_pile']['played_cards'])
    print("Pile score: %d" % state['current_round']['current_pile']['score'])
    print("Trump: %s" % (state['current_round']['trump']))
    print("===================== Hand =====================")
    print_card(state['current_round']['player']['hand'])
    print("===================== Legal Moves =====================")
    print_card(state['current_round']['legal_moves'])


def run_game():
    game = DiffGame(4, 2, FixedPredictinStrategy())

    state, _ = game.init_game()
    while not game.is_over():
        display_state(state)
        action = random.choice(state['current_round']['legal_moves'])
        print("===================== Plays =====================")
        print_card([action])
        state, _ = game.step(action)

    display_state(state)


run_game()
