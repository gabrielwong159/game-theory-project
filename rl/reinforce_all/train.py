import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import RMSprop
from tqdm import trange

from game import Game
from dqn import DQN
from plots import plot_rewards

GAMMA = 0.999
EPS = 0.05
SAVE_INTERVAL = 500

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
N_EPISODES = 200_000

N_PLAYERS = 5
LARGEST_CARD = 30
HAND_SIZE = 5
N_ROUNDS = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


def train():
    env = Game(N_PLAYERS, LARGEST_CARD, HAND_SIZE, N_ROUNDS)

    net = DQN(n_inputs=2*LARGEST_CARD, n_outputs=HAND_SIZE).to(device)
    optimizer = RMSprop(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    episodic_rewards = []
    for episode in trange(N_EPISODES):
        states, rewards, actions = generate_episode(env, net)
        optimize_model(net, optimizer, states, rewards, actions)
        
        episodic_rewards.append(sum(rewards[:, 0]))

        if episode % SAVE_INTERVAL == 0:
            torch.save(net.state_dict(), f'models/model_{episode}.pth')
        if episode % 100 == 0:
            plot_rewards(np.cumsum(episodic_rewards),
                         baseline=np.zeros_like(episodic_rewards))

    return episodic_rewards


def generate_episode(env, net):
    states, rewards, actions = [], [], []

    observation = env.reset()
    done = False
    
    while not done:
        state = torch.tensor(create_state(observation), dtype=torch.float, device=device)
        action = torch.tensor([select_action(net, state, observation.hands[i])
                                for i in range(N_PLAYERS)], dtype=torch.long, device=device)
        observation, reward, done, info = env.step(action.cpu().numpy())
        
        # previous state is appended here instead of new state
        # to maintain same number of states and actions
        states.append(state.cpu().numpy())
        rewards.append(reward)
        actions.append(action.cpu().numpy())
        
        if done:
            break
        else:
            state = torch.tensor(create_state(observation), dtype=torch.float, device=device)
    
    assert len(states) == len(rewards) == len(actions), [len(states), len(rewards), len(actions)]
    return np.asarray(states), np.asarray(rewards), np.asarray(actions)


def optimize_model(net, optimizer, states, rewards, actions):
    T = len(rewards)
    
    future_rewards = np.zeros([T, N_PLAYERS])
    future_rewards[-1] = rewards[-1]
    for i in range(T-2, -1, -1):
        future_rewards[i] = rewards[i] + GAMMA*future_rewards[i+1]
    future_rewards = torch.tensor(future_rewards, dtype=torch.float, device=device)

    states = torch.tensor(states, dtype=torch.float, device=device)
    action_probs = net(states)
    
    actions = torch.tensor(actions, dtype=torch.float, device=device)
    chosen_log_probs = Categorical(action_probs).log_prob(actions)
    
    # make discounts.shape == [HAND_SIZE, 1] for broadcasting onto [HAND_SIZE, N_PLAYERS]
    discounts = torch.tensor([[GAMMA**t] for t in range(T)], dtype=torch.float, device=device)
    loss = -torch.sum(discounts * future_rewards * chosen_log_probs) / N_PLAYERS
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

    
def create_state(observation):
    hands, history = observation
    state = []
    
    history = np.asarray(list(history), dtype=np.int)
    history_one_hot = np.zeros(LARGEST_CARD, dtype=np.float32)
    history_one_hot[history-1] = 1

    for i in range(N_PLAYERS):
        hand = hands[i]
        hand_one_hot = np.zeros(LARGEST_CARD, dtype=np.float32)
        hand_one_hot[hand-1] = 1
        state.append(np.concatenate([hand_one_hot, history_one_hot]))
    
    return np.asarray(state)


def select_action(net, state, hand):
    n_cards = len(hand)

    if np.random.rand() < EPS:
        action = np.random.randint(n_cards)
    else:
        with torch.no_grad():
            ordered_actions = torch.topk(net(state), k=HAND_SIZE, dim=1).indices.cpu().numpy()
            filtered_actions = ordered_actions[ordered_actions < n_cards]
            action = filtered_actions[0]
    return torch.tensor([[action]], dtype=torch.long, device=device)


if __name__ == '__main__':
    rewards = train()
    np.save('rewards', rewards)

