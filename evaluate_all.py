"""
Evaluate all four winners over 20 episodes each.
Produces honest average reward and failure rate statistics.
"""
import os
import pickle
import numpy as np
import neat
import gymnasium as gym

BASE_DIR = os.path.expanduser("~/Desktop/neat_bipedal_project")

EXPERIMENTS = [
    {
        "name":        "Baseline (speed)",
        "config":      os.path.join(BASE_DIR, "config-neat.txt"),
        "winner":      os.path.join(BASE_DIR, "results",
                                    "winner_genome.pkl"),
    },
    {
        "name":        "Efficiency",
        "config":      os.path.join(BASE_DIR, "config-neat.txt"),
        "winner":      os.path.join(BASE_DIR, "results_efficiency",
                                    "winner_efficiency.pkl"),
    },
    {
        "name":        "Stability",
        "config":      os.path.join(BASE_DIR, "config-neat.txt"),
        "winner":      os.path.join(BASE_DIR, "results_stability",
                                    "winner_stability.pkl"),
    },
    {
        "name":        "Ablation (fixed topology)",
        "config":      os.path.join(BASE_DIR, "config-neat-ablation.txt"),
        "winner":      os.path.join(BASE_DIR, "results_ablation",
                                    "winner_ablation.pkl"),
    },
]

NUM_EPISODES  = 20
FALL_THRESHOLD = 100  # reward below this = failed episode

print("\n" + "="*65)
print(f"{'Evaluating all winners — ' + str(NUM_EPISODES) + ' episodes each':^65}")
print("="*65)

results = []

for exp in EXPERIMENTS:
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        exp["config"]
    )

    with open(exp["winner"], "rb") as f:
        genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    rewards = []
    steps_list = []
    failures = 0

    for episode in range(NUM_EPISODES):
        env = gym.make("BipedalWalker-v3", render_mode=None)
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0

        for _ in range(1600):
            actions = np.clip(net.activate(obs), -1.0, 1.0)
            obs, reward, terminated, truncated, _ = env.step(actions)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        env.close()
        rewards.append(total_reward)
        steps_list.append(steps)
        if total_reward < FALL_THRESHOLD:
            failures += 1

    avg_reward   = np.mean(rewards)
    std_reward   = np.std(rewards)
    avg_steps    = np.mean(steps_list)
    failure_rate = (failures / NUM_EPISODES) * 100
    best_episode = max(rewards)
    worst_episode = min(rewards)

    results.append({
        "name":         exp["name"],
        "avg":          avg_reward,
        "std":          std_reward,
        "avg_steps":    avg_steps,
        "failure_rate": failure_rate,
        "best":         best_episode,
        "worst":        worst_episode,
    })

    print(f"\n{exp['name']}")
    print(f"  Avg reward:   {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  Best episode: {best_episode:.2f}")
    print(f"  Worst episode:{worst_episode:.2f}")
    print(f"  Avg steps:    {avg_steps:.0f}")
    print(f"  Failure rate: {failures}/{NUM_EPISODES} "
          f"({failure_rate:.0f}%)")

# ── Final comparison table ─────────────────────────────────────────────────
print("\n" + "="*65)
print(f"{'FINAL COMPARISON TABLE':^65}")
print("="*65)
print(f"{'Experiment':<28} {'Avg±Std':>14} {'Steps':>7} "
      f"{'Fails':>7} {'Best':>8}")
print("-"*65)
for r in results:
    print(f"{r['name']:<28} "
          f"{r['avg']:>7.1f}±{r['std']:<6.1f} "
          f"{r['avg_steps']:>7.0f} "
          f"{r['failure_rate']:>6.0f}% "
          f"{r['best']:>8.1f}")
print("="*65)
print("\nDone. Use these numbers in your report.")
