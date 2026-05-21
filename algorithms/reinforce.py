"""
Reinforce algorithim implementation 
(Monte-Carlo Policy Gradient) for cartpole

May 2026
"""

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")
