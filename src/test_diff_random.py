import random

from diff import DiffGame, FixedPredictionStrategy

from rlcard.utils.utils import print_card


TRAIN_CONFIG = {
    'players': 4,
    'rounds': 1,
    'prediction_strategy': FixedPredictionStrategy(157),
    'reward_strategy': 'default',
    'state_representation': 'compressed',
    'allow_step_back': False,
    'seed': random.randint(1, 999999)
}


def display_state(state: dict, payoff: list[float]) -> None:
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
    print("===================== Payoffs =====================")
    print([p * 157 for p in payoff])


def run_game():
    game = DiffGame()
    game.configure(TRAIN_CONFIG)

    state, _ = game.init_game()
    payoff = [0, 0, 0, 0]
    while not game.is_over():
        display_state(state, payoff)
        action = random.choice(state['current_round']['legal_moves'])
        print("===================== Plays =====================")
        print_card([action])
        state, _ = game.step(action.suit + action.rank)
        payoff = game.get_payoffs()

    display_state(state, payoff)


run_game()
