import numpy as np
from tqdm import tqdm
from env import *
from agents import *
from plots import *

BANDITS = 10
PROBLEMS = 2000
STEPS = 1000

def main(problems=PROBLEMS, steps=STEPS, stationary=True):
    eps = [0.1, 0.01, 0]
    agents = [UCBAgent(c=2), SampleAverageAgent(eps=0.1)]
    bandits = BANDITS
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

if __name__ == '__main__':
    main(stationary=False, problems=PROBLEMS, steps=STEPS)