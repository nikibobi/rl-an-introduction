#!/usr/bin/python
import argparse
import numpy as np
from math import log2
from tqdm import tqdm
from env import *
from agents import *
from plots import *

BANDITS = 10
PROBLEMS = 2000
STEPS = 1000

GROUPS = [
    {
        'label': r'$\varepsilon$-greedy',
        'param': r'$\varepsilon$',
        'interval': (1 / 128, 1 / 4),
        'factory': lambda eps: SampleAverageAgent(eps)
    },
    {
        'label': r'UCB',
        'param': r'$c$',
        'interval': (1 / 16, 4),
        'factory': lambda c: UCBAgent(c)
    },
    {
        'label': r'gradient bandit',
        'param': r'$\alpha$',
        'interval': (1 / 32, 2),
        'factory': lambda alpha: GradientBanditAgent(alpha)
    },
    {
        'label': r'greedy with optimistic initialization $\alpha=0.1$',
        'param': r'$Q_0$',
        'interval': (1 / 4, 4),
        'factory': lambda q0: ConstantStepAgent(eps=0, alpha=0.1, q0=q0)
    }
]

def experiment(agents, bandits, problems, steps, stationary, plot):
    env = create_bandits(bandits, stationary)
    total_rewards = np.zeros((steps, len(agents)))
    optimal_actions = np.zeros_like(total_rewards)
    for p in tqdm(range(problems)):
        env.reset()
        for agent in agents:
            agent.init(bandits)
        for t in range(steps):
            a_opt = env.optimal()
            for i, agent in enumerate(agents):
                a = agent.act()
                r = env.step(a)
                agent.update(a, r)
                total_rewards[t, i] += r
                optimal_actions[t, i] += a == a_opt
            env.update()

    optimal_actions /= problems

    if plot:
        legend = tuple(agent.legend() for agent in agents)
        plot_rewards(total_rewards, problems, steps, legend)
        plot_optimal(optimal_actions, steps, legend)

    last_rewards = total_rewards[-1, :] / problems
    return last_rewards

def experiments(groups, bandits, problems, steps, stationary, plot):
    results = []

    for group in groups:
        label, param, interval, factory = group.values()
        start, stop = map(lambda i: int(log2(i)), interval)
        args = [2 ** i for i in range(start, stop + 1)]
        agents = [factory(arg) for arg in args]
        average_rewards = experiment(agents, bandits, problems, steps, stationary, plot=False)
        results.append((args, average_rewards, label, param))

    if plot:
        plot_summary(results, steps)

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bandits', type=int)
    parser.add_argument('-n', '--problems', type=int)
    parser.add_argument('-t', '--steps', type=int)
    parser.add_argument('-s', '--stationary', dest='stationary', action='store_true')
    parser.add_argument('-S', '--non-stationary', dest='stationary', action='store_false')
    parser.add_argument('-p', '--plot', dest='plot', action='store_true')
    parser.add_argument('-P', '--no-plot', dest='plot', action='store_false')
    parser.set_defaults(bandits=BANDITS, problems=PROBLEMS, steps=STEPS, stationary=True, plot=True)
    args = vars(parser.parse_args())
    np.seterr(all='ignore')
    experiments(GROUPS, **args)

if __name__ == '__main__':
    main()