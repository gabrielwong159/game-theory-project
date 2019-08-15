import matplotlib.pyplot as plt


def plot_rewards(rewards, baseline):
    plt.figure(1)
    plt.clf()
    plt.plot(rewards, label='Actual')
    plt.plot(baseline, label='Baseline')

    for x in range(500, len(rewards), 500):
        plt.axvline(x, color='r', linestyle='--')

    plt.legend()
    plt.savefig('fig.png')
