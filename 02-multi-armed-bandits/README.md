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