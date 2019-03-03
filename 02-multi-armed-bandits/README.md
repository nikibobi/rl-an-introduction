# Experiments for Multi-Armed bandits

## Stationary problem

Tests are made with action value sample-average agents for ε = `0.1`, `0.01`, `0.001` and `0`
![stationary total rewards](results/stationary_total_rewards.png)
![stationary average rewards](results/stationary_average_rewards.png)
![stationary optimal actions](results/stationary_optimal_actions.png)

## Non-stationary problem

Tests made with two agents:
 - Agent with standard sample-average action value with ε = `0.1`
 - Agent with fixed step-size α = `0.1` and ε = `0.1`

![nonstationary total rewards](results/nonstationary_total_rewards.png)
![nonstationary average rewards](results/nonstationary_average_rewards.png)

We can see that the fixed-step agent performed better for a non-stationary problem.

## ucb Initial Values

Two constant step agents with α = `0.1`:
- One greedy (ε = `0`) and optimistic initial value = `5`
- One ε-greedy (ε = `0.1`)

![optimistic total rewards](results/optimistic_total_rewards.png)
![optimistic average rewards](results/optimistic_average_rewards.png)
![optimistic optimal actions](results/optimistic_optimal_actions.png)

It is clear that the optimistic initial values agent performs better in the long run for stationary problems.

## Upper-Confidence-Bound

![ucb total rewards](results/ucb_total_rewards.png)
![ucb average rewards](results/ucb_average_rewards.png)
![ucb optimal actions](results/ucb_optimal_actions.png)


## Gradient Bandits with baseline

![gradient bandits rewards](results/gradient_rewards.png)
![gradient bandits optimal actions](results/gradient_optimal_actions.png)

## Summary

Summary of several different agents over a range of hyperparameters

![summary average rewards](results/summary.png)