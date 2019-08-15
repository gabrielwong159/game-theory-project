import numpy as np
import torch
from .dqn import DQN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Agent(object):
    def __init__(self, largest_card, hand_size, model_path=None):
        self.largest_card = largest_card
        self.hand_size = hand_size
        
        self.net = DQN(largest_card*2, hand_size).to(device)
        if model_path:
            self.net.load_state_dict(torch.load(model_path))
        self.net.eval()

    def create_state(self, hand, history):
        hand_one_hot = np.zeros(self.largest_card, dtype=np.float32)
        hand_one_hot[hand-1] = 1

        history = np.asarray(list(history), dtype=np.int64)
        history_one_hot = np.zeros(self.largest_card, dtype=np.float32)
        history_one_hot[history-1] = 1

        return np.concatenate([hand_one_hot, history_one_hot])

    def select_action(self, hand, history):
        state = self.create_state(hand, history)
        state = torch.tensor([state], dtype=torch.float, device=device)
        
        with torch.no_grad():
            probs = self.net(state)
        ordered_actions = torch.topk(probs, k=self.hand_size, dim=1).indices.cpu().numpy()

        n_cards = len(hand)
        filtered_actions = ordered_actions[ordered_actions < n_cards]
        action = filtered_actions[0]
        return action
