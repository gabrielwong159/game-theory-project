import numpy as np
from enum import Enum
from .strategy import Strategy


class Choosing(Enum):
    RANDOM, BEST, BEST_VS_RANDOM, RL_VS_RANDOM = range(4)


class Game(object):
    def __init__(self, n_players, largest_card, hand_size, n_rounds, agent=None):
        self.n_players = n_players
        self.largest_card = largest_card
        self.hand_size = hand_size
        self.n_rounds = n_rounds
        
        self.WIN_REWARD = 1
        self.LOSE_REWARD = -1
        
        self.choosing = Choosing.RL_VS_RANDOM
        print(f'Choosing cards by {self.choosing}')
        
        self.strategy = Strategy(n_players, largest_card, agent)
    
    def play_game(self):
        hands = self.sample_hands()
        history = set()
        scores = np.zeros([self.n_players, self.n_rounds], np.int)
        
        for i in range(self.n_rounds):
            hands, history, winner, loser = self.play_round(hands, history)
            scores[winner, i] += self.WIN_REWARD
            scores[loser, i] += self.LOSE_REWARD

        return scores

    def sample_hands(self):
        n_cards = self.n_players * self.hand_size
        cards = np.random.choice(range(1, self.largest_card+1),
                                 size=n_cards, replace=False)
        hands = cards.reshape(self.n_players, -1)
        return hands

    def play_round(self, hands, history):
        card_indices = self.pick_cards(hands, history)
        chosen_cards = hands[range(self.n_players), card_indices]
        history.update(chosen_cards)
        
        winner, loser = self.get_winner_and_loser(chosen_cards)
        hands = self.remove_from_hands(hands, card_indices)

        return hands, history, winner, loser
    
    def pick_cards(self, hands, history):
        if self.choosing == Choosing.RANDOM:
            return self.strategy.choose_random(hands, history)
        elif self.choosing == Choosing.BEST:
            return self.strategy.choose_best(hands, history)
        elif self.choosing == Choosing.BEST_VS_RANDOM:
            return self.strategy.best_vs_random(hands, history)
        elif self.choosing == Choosing.RL_VS_RANDOM:
            return self.strategy.rl_vs_random(hands, history)
        else:
            raise

    def get_winner_and_loser(self, chosen_cards):
        sorted_players = np.argsort(chosen_cards)  # ascending order
        winner, loser = sorted_players[[-2, -1]]
        return winner, loser

    def remove_from_hands(self, hands, card_indices):
        mask = np.ones(hands.shape, dtype=np.bool)
        mask[range(self.n_players), card_indices] = False
        hands = hands[mask].reshape(self.n_players, -1)
        return hands
