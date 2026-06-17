"""
DQN (Deep Q-Network) algorithm
Core Idea:
DQN fixes the instability of using neural nets to approximate Q-values
i.e. "learn the value of actions, but stabilize training so it doesnt explode"

This is done via two tricks:
    Experience Replay  — store (s, a, r, s') tuples in a buffer, sample randomly
    Target Network     — a frozen copy of Q that updates slowly, not every step

Instead of optimizing a policy directly, we learn Q(s,a):
    the expected total reward for taking action a in state s

Then act greedily (with some randomness via ε-greedy):
    a = argmax_a Q(s, a)     with probability 1 - ε
    a = random               with probability ε

LOOP:
1. Take a step using ε-greedy policy, store (s, a, r, s', done) in replay buffer
2. Sample a random minibatch from the buffer
3. Compute TD target using the frozen target network
4. Update Q-network to minimize Bellman error
5. Every C steps, copy Q-network weights → target network
6. Decay ε over time (less random as training matures)
7. Repeat

Math:
TD Target (what Q should predict):
    y = r + γ * max_a' Q_target(s', a')     if not done
    y = r                                    if done

Loss (Bellman error):
    L = mean( (y - Q(s, a))² )

ε-greedy decay:
    ε = max(ε_min, ε_start * ε_decay^step)
"""