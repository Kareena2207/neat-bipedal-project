import gymnasium as gym
import numpy as np

env = gym.make("BipedalWalker-v3", render_mode="human")
obs, _ = env.reset()

print("=== Observation breakdown ===")
labels = [
    "Hull angle", "Hull angular velocity",
    "Horizontal velocity (vx)", "Vertical velocity (vy)",
    "Hip 1 angle", "Hip 1 speed",
    "Knee 1 angle", "Knee 1 speed", "Leg 1 ground contact",
    "Hip 2 angle", "Hip 2 speed",
    "Knee 2 angle", "Knee 2 speed", "Leg 2 ground contact",
    "Lidar 0", "Lidar 1", "Lidar 2", "Lidar 3", "Lidar 4",
    "Lidar 5", "Lidar 6", "Lidar 7", "Lidar 8", "Lidar 9"
]
for i, (label, val) in enumerate(zip(labels, obs)):
    print(f"  obs[{i:2d}] {label}: {val:.4f}")

# Try random actions to see what happens
for step in range(200):
    action = env.action_space.sample()  # completely random motors
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step}: reward={reward:.3f}")
    if terminated or truncated:
        break

env.close()
