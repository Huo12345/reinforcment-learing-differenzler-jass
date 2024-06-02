import torch
import numpy as np
from numpy import ndarray


class DqnAgent:
    """
    Agent that uses a DQN to decide which action to take.
    """

    def __init__(self, path: str, device: torch.device = torch.device('cpu')):
        """
        Loads a pretrained DQN model.

        :param path: Path to the saved model.
        :param device: Torch device to use.
        """
        dqn_agent = torch.load(path, map_location=device)
        self.model = dqn_agent.q_estimator.qnet
        self.model.eval()
        self.device = device
        self.use_raw = False

    def step(self, state: dict) -> int:
        """
        Calculates the next step to be taken based on the partial observation for the current player.

        :param state: Observation of the current game state from the perspective of the current player.
        :return: The action to take.
        """
        q_values = self._predict(state)
        legal_actions = list(state['legal_actions'].keys())
        action_idx = legal_actions.index(np.argmax(q_values))

        return legal_actions[action_idx]

    def eval_step(self, state: dict) -> tuple[int, dict]:
        """
        Calculates the next step to be taken based on the partial observation for the current player.

        :param state: Observation of the current game state from the perspective of the current player.
        :return: Tuple containing the action to take and infos to analyse the decision.
        """
        q_values = self._predict(state)
        legal_actions = list(state['legal_actions'].keys())
        action_idx = legal_actions.index(np.argmax(q_values))

        info = {'values': {state['raw_legal_actions'][i]: float(q_values[list(state['legal_actions'].keys())[i]]) for i
                           in range(len(state['legal_actions']))}}

        return legal_actions[action_idx], info

    def _predict(self, state: dict) -> ndarray:
        """
        Calculates the masked q values based on the trained q network and the legal actions.

        :param state: Observation of the current game state from the perspective of the current player.
        :return: Masked q values, containing -infinity if the move is illegal, otherwise network prediction.
        """
        with torch.no_grad():
            # Reshaping into single batch shape
            obs = np.expand_dims(state['obs'], 0)

            # Calculating q values
            s = torch.from_numpy(obs).float().to(self.device)
            q = self.model(s).cpu().detach().numpy()[0]

            # Masking q values
            masked_q = -np.inf * np.ones(len(q), dtype=float)
            legal_actions = list(state['legal_actions'].keys())
            masked_q[legal_actions] = q[legal_actions]
            return masked_q
