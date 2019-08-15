import matplotlib.pyplot as plt


def plot_cum_scores(cum_scores, title: str):
    # cum_scores should be of shape [N_PLAYERS, N_GAMES]
    assert cum_scores.ndim == 2

    n_players = len(cum_scores)
    for i in range(n_players):
        plt.plot(cum_scores[i], label=f'Player {i}')
    
    plt.xlabel('Game number')
    plt.ylabel('Cumulative reward')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_reward_freq(scores, player_index, round_index):
    # scores should of shape [N_PLAYERS, N_GAMES, N_ROUNDS]
    assert scores.ndim == 3

    plt.hist(scores[player_index, :, round_index])
    plt.xlabel('Rewards')
    plt.ylabel('Frequency')
    plt.title('Frequency of rewards')
    plt.show()
