import neat
import numpy as np
import gymnasium as gym

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    "config-neat.txt"
)

# ── Fix: create genome through Population so innovation tracker is set ────
# Instead of manually calling configure_new(), we let NEAT create a full
# population and just grab the first genome from it. This ensures all
# internal trackers are properly initialized.
population = neat.Population(config)

# Grab the first genome from the population (any genome will do for testing)
first_genome = list(population.population.values())[0]

print("=== Random genome structure ===")
hidden = [n for n in first_genome.nodes if n >= 0]
print(f"  Hidden nodes:  {len(hidden)}")
print(f"  Connections:   {len(first_genome.connections)}")
print(f"  Enabled conns: {sum(1 for c in first_genome.connections.values() if c.enabled)}")

# ── Build neural network from the genome ──────────────────────────────────
net = neat.nn.FeedForwardNetwork.create(first_genome, config)

# ── Run it in BipedalWalker ───────────────────────────────────────────────
env = gym.make("BipedalWalker-v3", render_mode=None)
obs, _ = env.reset()

total_reward = 0.0
step_count   = 0

for step in range(1600):
    raw_outputs = net.activate(obs)
    actions     = np.clip(raw_outputs, -1.0, 1.0)
    obs, reward, terminated, truncated, _ = env.step(actions)
    total_reward += reward
    step_count   += 1
    if terminated or truncated:
        break

env.close()

print(f"\n=== Fitness result ===")
print(f"  Steps survived: {step_count}")
print(f"  Total reward:   {total_reward:.4f}")
print(f"  Fell?           {'Yes (-100 penalty)' if total_reward < -90 else 'No'}")
print(f"\nThis genome's fitness = {total_reward:.4f}")
print("(NEAT will try to MAXIMIZE this number over 200 generations)")
