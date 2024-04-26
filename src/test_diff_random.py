import random

from diff import DiffGame, FixedPredictionStrategy

from rlcard.utils.utils import print_card


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
    game = DiffGame()
    game.configure({
        'players': 4,
        'rounds': 2,
        'prediction_strategy': FixedPredictionStrategy(157 // 4)
    })

    state, _ = game.init_game()
    while not game.is_over():
        display_state(state)
        action = random.choice(state['current_round']['legal_moves'])
        print("===================== Plays =====================")
        print_card([action])
        state, _ = game.step(action)

    display_state(state)


run_game()
