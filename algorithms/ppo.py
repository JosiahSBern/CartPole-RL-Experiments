"""
PPO (Proximal Policy Optimization) algorithm
Core Idea:
PPO fixes the issue of taking too drastic of a step from policy to policy
i.e. "update the polciy, but dont change it too much in a single step"

This done by measuring a ration:
    ratio = π_new(a|s) / π_old(a|s)
Then CLIPS this ratio to be [0.8,1.2] or can never be more than a 20% per policy

LOOP:
1. Collect N steps of experience using current policy
2. Compute advantages
3. For K epochs, reuse that same data to update the network (Key differences)
4. Clip the update if the policy tries to change too much
5. Repeat

Math:
Clipped Loss
    ratio   = exp(log π_new - log π_old)
    L_clip =  L_CLIP  = -mean( min(ratio * A,  clip(ratio, 1-ε, 1+ε) * A) )

Full loss (3 terms):
    L = L_CLIP           policy update (clipped)
      + 0.5 * L_value    critic accuracy (MSE)
      - 0.01 * entropy   exploration bonus (stops policy collapsing)
"""
