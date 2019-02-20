import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from scipy.stats import norm
from tqdm import tqdm
from abc import abstractmethod, ABC

BANDITS = 10
PROBLEMS = 2000
STEPS = 1000

def generate_stationary(n=BANDITS):
    means = np.random.randn(n)
    return means

def generate_nonstationary(n=BANDITS):
    mean = np.random.randn()
    means = np.full(n, mean)
    return means

def update_nonstationary(means):
    means += np.random.normal(scale=0.01, size=means.shape)

def plot_bandits(means, normal):
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

class Agent(ABC):
    def __init__(self, eps, q0=0):
        self.eps = eps
        self.q0 = q0
    
    def init(self, actions):
        self.actions = actions
        self.q = np.full(actions, self.q0, dtype=np.float)
    
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

class SampleAverageAgent(Agent):
    def __init__(self, eps, q0=0):
        super().__init__(eps, q0)
    
    def init(self, actions):
        super().init(actions)
        self.n = np.zeros(actions)

    def step_size(self, a):
        return 1 / self.n[a]
    
    def update(self, a, r):
        self.n[a] += 1
        super().update(a, r)
    
    def legend(self):
        return r'$\epsilon = {}, Q_0 = {}$ Sample Average'.format(self.eps, self.q0)

class ConstantStepAgent(Agent):
    def __init__(self, eps, alpha, q0=0):
        super().__init__(eps, q0)
        self.alpha = alpha

    def step_size(self, a):
        return self.alpha
    
    def legend(self):
        return r'$\epsilon = {}, \alpha = {}, Q_0 = {}$ Constant Step'.format(self.eps, self.alpha, self.q0)

class UCBAgent(SampleAverageAgent):
    def __init__(self, c):
        super().__init__(eps=0)
        self.c = c

    def act(self):
        return np.argmax(self.q + self.c * np.sqrt(np.log(np.sum(self.n)) / self.n))

    def legend(self):
        return r'$c = {}$ Upper-Confidence-Bound'.format(self.c)

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

def normal_values(num, e=0.01):
    return norm.ppf(np.linspace(e, 1 - e, num=num, dtype=np.float))

def main(problems=PROBLEMS, steps=STEPS, stationary=True):
    eps = [0.1, 0.01, 0]
    normal = normal_values(100)
    agents = [UCBAgent(c=2), SampleAverageAgent(eps=0.1)]
    total_rewards = np.zeros((steps, len(agents)))
    optimal_actions = np.zeros_like(total_rewards)
    for p in tqdm(range(problems)):
        bandits = BANDITS
        if stationary:
            means = generate_stationary(bandits)
        else:
            means = generate_nonstationary(bandits)
        bandit = lambda a: np.random.randn() + means[a]
        for agent in agents:
            agent.init(bandits)
        for t in range(steps):
            for i, agent in enumerate(agents):
                a_opt = np.argmax(means)
                a = agent.act()
                r = bandit(a)
                agent.update(a, r)
                total_rewards[t, i] += r
                optimal_actions[t, i] += a == a_opt
            if not stationary:
                update_nonstationary(means)

    optimal_actions /= problems
    legend = tuple(agent.legend() for agent in agents)
    plot_rewards(total_rewards, problems, steps, legend)
    plot_optimal(optimal_actions, steps, legend)

if __name__ == '__main__':
    main(stationary=True, problems=PROBLEMS, steps=STEPS)