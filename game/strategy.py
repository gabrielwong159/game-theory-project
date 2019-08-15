import numpy as np
from enum import Enum


class Scoring(Enum):
    PROB, SOLN = range(2)


class Strategy(object):
    def __init__(self, n_players, largest_card):
        self.n_players = n_players
        self.largest_card = largest_card
        
        self.scoring = Scoring.SOLN
        print(f'Scoring by {self.scoring}')

    def choose_random(self, hands, history):
        n_players, n_cards_remaining = hands.shape
        return np.random.choice(range(n_cards_remaining),
                                size=n_players)

    def choose_best(self, hands, history):
        n_players, n_cards_remaining = hands.shape
        hand_scores = np.zeros(hands.shape)

        for i in range(n_players):
            unseen_cards = self.get_unseen_cards(history, hands[i])
            for j, card in enumerate(hands[i]):
                hand_scores[i, j] = self.score(card, unseen_cards)
        return hand_scores.argmax(axis=1)
    
    def best_vs_random(self, hands, history):
        # all other players are random
        card_indices = self.choose_random(hands, history)

        # choose best action for player 0
        n_players, n_cards_remaining = hands.shape
        unseen_cards = self.get_unseen_cards(history, hands[0])
        scores = [self.score(card, unseen_cards) for card in hands[0]]
        card_indices[0] = np.argmax(scores)

        return card_indices
        
    def get_unseen_cards(self, history, current_hand):
        seen_cards = history | set(current_hand)
        unseen_cards = set(range(1, self.largest_card+1)) - seen_cards
        return unseen_cards

    def score(self, card, unseen_cards):
        if self.scoring == Scoring.PROB:
            return self.score_prob(card, unseen_cards)
        elif self.scoring == Scoring.SOLN:
            return self.score_soln(card, unseen_cards)
        else:
            raise

    def score_prob(self, card, unseen_cards):
        # probability of having (n-1) cards be (n-2) smaller and 1 larger
        num_lower = sum(c < card for c in unseen_cards)
        p = num_lower / len(unseen_cards)  # F(x)
        return p**(self.n_players-2) * (1 - p) - p**(self.n_players-1)

    def score_soln(self, card, unseen_cards):
        # use exact solution of playing x = (n-2) / (2n-2), given [0, 1]
        ratio = (self.n_players - 2) / (2*self.n_players - 2)
        target = np.percentile(list(unseen_cards), ratio*100)
        return -np.abs(card - target)  # maximize the negative abs. diff.
