import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from scipy.stats import norm
from tqdm import tqdm

BANDITS = 10
MIN_BANDITS = 5
MAX_BANDITS = 15
PROBLEMS = 2000
STEPS = 1000

def generate(normal, n=BANDITS, plot=False):
    means = np.random.randn(n)
    data = np.repeat(normal[:, np.newaxis], n, axis=1) + means

    if plot:
        sns.set(style='whitegrid')
        ax = sns.violinplot(data=data, inner=None)
        ax = sns.scatterplot(data=means, ax=ax)
        ax.set_xlabel('Actions')
        ax.set_ylabel('Q(a)')
        ax.set_xticklabels(np.arange(n) + 1)
        ax.yaxis.set_major_locator(ticker.MultipleLocator())
        plt.show()

    return means

class Agent(object):
    def __init__(self, eps):
        self.eps = eps
    
    def init(self, actions):
        self.actions = actions
        self.n = np.zeros(actions)
        self.q = np.zeros(actions)
    
    def act(self):
        if np.random.rand() > self.eps:
            return np.argmax(self.q)
        else:
            return np.random.choice(self.actions)
    
    def update(self, a, r):
        self.n[a] += 1
        self.q[a] += 1/self.n[a] * (r - self.q[a])

def plot_rewards(metric, title, eps):
    plt.plot(metric)
    plt.title(title)
    plt.gcf().canvas.set_window_title(title)
    plt.xlim(0, STEPS)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.gca().legend(tuple('ε = {}'.format(e) for e in eps))
    plt.savefig('results/{}.png'.format(title.lower().replace(' ', '_')))
    plt.show()

def main(problems=PROBLEMS, steps=STEPS):
    eps = [0.1, 0.01, 0.001, 0]
    normal = np.random.randn(10000)
    agents = [Agent(eps=e) for e in eps]
    total_rewards = np.zeros((steps, len(agents)))
    for p in tqdm(range(problems)):
        bandits = BANDITS #np.random.randint(MIN_BANDITS, MAX_BANDITS + 1)
        means = generate(normal, bandits, plot=False)
        bandit = lambda a: np.random.randn() + means[a]
        for agent in agents:
            agent.init(bandits)
        for t in range(steps):
            for i, agent in enumerate(agents):
                a = agent.act()
                r = bandit(a)
                agent.update(a, r)
                total_rewards[t, i] += r
    
    average_rewards = total_rewards / problems
    plot_rewards(total_rewards, 'Total Rewards', eps)
    plot_rewards(average_rewards, 'Average Rewards', eps)

if __name__ == '__main__':
    main(problems=PROBLEMS)