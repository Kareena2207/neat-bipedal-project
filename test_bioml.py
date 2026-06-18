import neat
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Test that BipedalWalker loads
env = gym.make("BipedalWalker-v3")
obs, _ = env.reset()
print(f"Observation shape: {obs.shape}")   # Should print: (24,)
print(f"Action space: {env.action_space}") # Should print: Box(-1, 1, (4,), float32)
env.close()
print("All good!")
