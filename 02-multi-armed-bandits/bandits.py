import numpy as np
from tqdm import tqdm
from plots import *
from agents import *

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

def update_nonstationary(means, var=0.01):
    means += np.random.normal(scale=var, size=means.shape)

def main(problems=PROBLEMS, steps=STEPS, stationary=True):
    eps = [0.1, 0.01, 0]
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