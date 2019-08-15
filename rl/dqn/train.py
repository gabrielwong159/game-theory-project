import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import RMSprop
from tqdm import trange

from game import Game
from dqn import DQN
from memory import ReplayMemory, Transition
from plots import plot_rewards

GAMMA = 0.99
EPS_START = 0.2
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 30
SAVE_INTERVAL = 500

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
MEMORY_SIZE = 50_000
N_EPISODES = 10_000

N_PLAYERS = 5
LARGEST_CARD = 30
HAND_SIZE = 5
N_ROUNDS = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


def train():
    policy_net = DQN(n_inputs=2*LARGEST_CARD, n_outputs=HAND_SIZE).to(device)
    target_net = DQN(n_inputs=2*LARGEST_CARD, n_outputs=HAND_SIZE).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = RMSprop(policy_net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    memory = ReplayMemory(MEMORY_SIZE)

    env = Game(N_PLAYERS, LARGEST_CARD, HAND_SIZE, N_ROUNDS)
    select_action = generate_action_selector()

    rewards = []
    for episode in trange(N_EPISODES):
        total_reward = 0
        observation = env.reset()
        done = False

        while not done:
            state = torch.tensor([create_state(observation)], dtype=torch.float, device=device)
            action = select_action(policy_net, state, observation.hand)

            observation, reward, done, info = env.step(action.item())
            total_reward += reward
            
            if not done:
                next_state = torch.tensor([create_state(observation)], dtype=torch.float, device=device)
            else:
                next_state = None
            reward = torch.tensor([reward], device=device)
            memory.push(state, action, next_state, reward)
            state = next_state
            
            optimize_model(policy_net, target_net, optimizer, memory)
            if done:
                rewards.append(total_reward)
                break
        
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if episode % SAVE_INTERVAL == 0:
            torch.save(target_net.state_dict(), f'models/model_{episode}.pth')
        if episode % 100 == 0:
            plot_rewards(np.cumsum(rewards), baseline=np.zeros(len(rewards)))

    return rewards


def optimize_model(policy_net, target_net, optimizer, memory):
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # compute a mask of non-final states
    non_final_mask = torch.tensor([s is not None for s in batch.next_state],
                                  device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # model computes Q(s_t)
    # use this to compute Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # V(s_{t+1})
    # all final states have 0 value
    # double Q-learning implemented
    policy_best_actions = policy_net(non_final_next_states).argmax(dim=1)
    i = torch.arange(len(policy_best_actions))
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states)[i, policy_best_actions].detach()

    # expected Q values
    expected_state_action_values = reward_batch + (next_state_values * GAMMA)
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(dim=1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    
def create_state(observation):
    hand, history = observation

    hand_one_hot = np.zeros(LARGEST_CARD, dtype=np.float32)
    hand_one_hot[hand-1] = 1
    
    history = np.asarray(list(history), dtype=np.int)
    history_one_hot = np.zeros(LARGEST_CARD, dtype=np.float32)
    history_one_hot[history-1] = 1
    
    return np.concatenate([hand_one_hot, history_one_hot])


def generate_action_selector():
    steps_done = 0

    def select_action(net, state, hand):
        nonlocal steps_done
        steps_done += 1
        
        n_cards = len(hand)
        
        eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-steps_done / EPS_DECAY)
        if np.random.rand() < eps_threshold:
            action = np.random.randint(n_cards)
        else:
            with torch.no_grad():
                ordered_actions = torch.topk(net(state), k=HAND_SIZE, dim=1).indices.cpu().numpy()
                filtered_actions = ordered_actions[ordered_actions < n_cards]
                action = filtered_actions[0]
        return torch.tensor([[action]], dtype=torch.long, device=device)
    
    return select_action


if __name__ == '__main__':
    rewards = train()
    np.save('rewards', rewards)
