#!/usr/bin/python
import argparse
import numpy as np
from tqdm import tqdm
from env import *
from agents import *
from plots import *

BANDITS = 10
PROBLEMS = 2000
STEPS = 1000

def experiment(bandits, problems, steps, stationary):
    eps = [0.1, 0.01, 0]
    agents = [GradientBanditAgent(alpha=0.1), GradientBanditAgent(alpha=0.4)]
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
    legend = tuple(agent.legend() for agent in agents)
    plot_rewards(total_rewards, problems, steps, legend)
    plot_optimal(optimal_actions, steps, legend)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bandits', type=int)
    parser.add_argument('-n', '--problems', type=int)
    parser.add_argument('-t', '--steps', type=int)
    parser.add_argument('-s', '--stationary', dest='stationary', action='store_true')
    parser.add_argument('-S', '--non-stationary', dest='stationary', action='store_false')
    parser.set_defaults(bandits=BANDITS, problems=PROBLEMS, steps=STEPS, stationary=True)
    args = vars(parser.parse_args())
    experiment(**args)

if __name__ == '__main__':
    main()