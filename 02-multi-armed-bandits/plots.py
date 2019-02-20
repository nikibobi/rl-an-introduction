import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from scipy.stats import norm

def generate_normal(num=100, e=0.01):
    return norm.ppf(np.linspace(e, 1 - e, num=num, dtype=np.float))

normal = generate_normal()

def plot_bandits(means):
    n = len(means)
    data = np.repeat(normal[:, np.newaxis], n, axis=1) + means
    sns.set(style='whitegrid')
    ax = sns.violinplot(data=data, inner=None)
    ax = sns.scatterplot(data=means, ax=ax)
    ax.set_xlabel('Actions')
    ax.set_ylabel('Q(a)')
    ax.set_xticklabels(np.arange(n) + 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator())
    plt.show()

def plot_rewards(metric, n, xlim, legend, filename='rewards', title='Rewards'):
    fig, ax1 = plt.subplots()
    fig.canvas.set_window_title(title)
    ax1.set_title(title)
    ax1.plot(metric)
    ax1.legend(legend)
    ax1.set_xlim(0, xlim)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Total Reward')
    ax2 = ax1.twinx()
    ax2.plot(metric / n)
    ax2.set_ylabel('Average Reward')
    fig.tight_layout()
    fig.savefig('results/{}.png'.format(filename))
    plt.show()

def plot_optimal(metric, xlim, legend, filename='optimal_actions', title='Optimal Actions'):
    fig, ax1 = plt.subplots()
    fig.canvas.set_window_title(title)
    ax1.set_title(title)
    ax1.plot(metric * 100.0)
    ax1.legend(legend)
    ax1.set_xlim(0, xlim)
    ax1.set_xlabel('Step')
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('Optimal Action %')
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax2 = ax1.twinx()
    ax2.plot(metric)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Optimal Action P')
    fig.tight_layout()
    fig.savefig('results/{}.png'.format(filename))
    plt.show()