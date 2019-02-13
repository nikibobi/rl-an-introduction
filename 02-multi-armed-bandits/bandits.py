import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from scipy.stats import norm
from tqdm import tqdm
from abc import abstractmethod

BANDITS = 10
PROBLEMS = 2000
STEPS = 1000

def generate_stationary(normal, n=BANDITS, plot=False):
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

def generate_nonstationary(n=BANDITS):
    mean = np.random.randn()
    means = np.full(n, mean)

    return means

def update_nonstationary(means):
    means += np.random.normal(scale=0.01, size=means.shape)

class Agent(object):
    def __init__(self, eps):
        self.eps = eps
    
    def init(self, actions):
        self.actions = actions
        self.q = np.zeros(actions)
    
    def act(self):
        if np.random.rand() > self.eps:
            return np.argmax(self.q)
        else:
            return np.random.choice(self.actions)
    
    @abstractmethod
    def step_size(self, a):
        pass

    def update(self, a, r):
        self.q[a] += self.step_size(a) * (r - self.q[a])

    @abstractmethod
    def legend(self):
        pass

class ActionValueAgent(Agent):
    def __init__(self, eps):
        super().__init__(eps)
    
    def init(self, actions):
        super().init(actions)
        self.n = np.zeros(actions)

    def step_size(self, a):
        return 1 / self.n[a]
    
    def update(self, a, r):
        self.n[a] += 1
        super().update(a, r)
    
    def legend(self):
        return 'ε = {}, Action Value'.format(self.eps)

class FixedStepAgent(Agent):
    def __init__(self, eps, alpha):
        super().__init__(eps)
        self.alpha = alpha

    def step_size(self, a):
        return self.alpha
    
    def legend(self):
        return 'ε = {}, α = {}, Fixed Step'.format(self.eps, self.alpha)


def plot_rewards(metric, title, xlim, legend):
    plt.plot(metric)
    plt.title(title)
    plt.gcf().canvas.set_window_title(title)
    plt.xlim(0, xlim)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.gca().legend(legend)
    plt.savefig('results/{}.png'.format(title.lower().replace(' ', '_')))
    plt.show()

def main(problems=PROBLEMS, steps=STEPS, stationary=True):
    eps = [0.1, 0.01, 0.001, 0]
    normal = np.random.randn(10000)
    agents = [ActionValueAgent(eps=0.1), FixedStepAgent(eps=0.1, alpha=0.1)]
    total_rewards = np.zeros((steps, len(agents)))
    for p in tqdm(range(problems)):
        bandits = BANDITS
        if stationary:
            means = generate_stationary(normal, bandits, plot=False)
        else:
            means = generate_nonstationary(bandits)
        bandit = lambda a: np.random.randn() + means[a]
        for agent in agents:
            agent.init(bandits)
        for t in range(steps):
            for i, agent in enumerate(agents):
                a = agent.act()
                r = bandit(a)
                agent.update(a, r)
                total_rewards[t, i] += r
            if not stationary:
                update_nonstationary(means)
    
    average_rewards = total_rewards / problems
    legend = tuple(agent.legend() for agent in agents)
    plot_rewards(total_rewards, 'Total Rewards', steps, legend)
    plot_rewards(average_rewards, 'Average Rewards', steps, legend)

if __name__ == '__main__':
    main(stationary=False, problems=PROBLEMS, steps=10_000)