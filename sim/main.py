import numpy as np
from tqdm import trange

from game import Game
from rl import Agent
from plots import plot_cum_scores, plot_reward_freq

N_PLAYERS = 5
LARGEST_CARD = 30
HAND_SIZE = 5
N_ROUNDS = 5
N_GAMES = 10_000

GAMMA = 0.5

np.random.seed(0)


def main():
    model_path = '../rl/reinforce/models/model_045500_537.pth'
    agent = Agent(LARGEST_CARD, HAND_SIZE, model_path)
    game = Game(N_PLAYERS, LARGEST_CARD, HAND_SIZE, N_ROUNDS, agent)

    scores_per_game = []
    for i in trange(N_GAMES):
        scores = game.play_game()
        scores_per_game.append(scores)

    # initially [n_games, n_players, n_rounds]
    weights = [GAMMA**i for i in range(N_ROUNDS)]
    scores_per_player_per_game = np.asarray(scores_per_game).transpose(1, 0, 2)
    scores_per_player = np.average(scores_per_player_per_game, axis=2, weights=weights) * sum(weights)
    cum_scores = np.cumsum(scores_per_player, axis=1)
    
    plot_cum_scores(cum_scores, title='Strategy: best vs. random')
    plot_reward_freq(scores_per_player_per_game,
                     player_index=0, round_index=-1)


if __name__ == '__main__':
    main()
