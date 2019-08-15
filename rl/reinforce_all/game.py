import numpy as np
from collections import namedtuple


Observation = namedtuple('Observation', 'hands history')
Info = namedtuple('Info', 'hands cards_played')


class Game(object):
    def __init__(self, n_players, largest_card, hand_size, n_rounds):
        self.n_players = n_players
        self.largest_card = largest_card
        self.hand_size = hand_size
        self.n_rounds = n_rounds
        
        self.PLAYER_INDEX = 0
        self.WIN_REWARD = 1.0
        self.LOSE_REWARD = -1.0
        
    def reset(self):
        self.hands = self.generate_hands()
        self.history = set()
        self.round_number = 0
        return self.get_observation()
    
    def step(self, actions):
        chosen_cards = self.hands[range(self.n_players), actions]
        self.update_hands(actions)

        history = self.history.update(chosen_cards)
        reward = self.get_reward(chosen_cards)
        
        self.round_number += 1
        done = self.round_number == self.n_rounds
        
        observation = self.get_observation()
        info = Info(hands=self.hands, cards_played=chosen_cards)
        return observation, reward, done, info

    def get_reward(self, chosen_cards):
        sorted_players = np.argsort(chosen_cards)
        rewards = np.zeros(self.n_players, dtype=np.float32)
        rewards[sorted_players[-2]] = self.WIN_REWARD
        rewards[sorted_players[-1]] = self.LOSE_REWARD
        return rewards

    def update_hands(self, card_indices):
        mask = np.ones(self.hands.shape, dtype=np.bool)
        mask[range(self.n_players), card_indices] = False
        self.hands = self.hands[mask].reshape(self.n_players, -1)
    
    def get_observation(self):
        return Observation(hands=self.hands, history=self.history)

    def generate_hands(self):
        n_cards = self.n_players * self.hand_size
        cards = np.random.choice(range(1, self.largest_card+1),
                                 size=n_cards, replace=False)
        hands = cards.reshape(self.n_players, -1)
        return hands
