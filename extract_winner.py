"""
Scans every checkpoint, finds the single best genome across all of them,
saves it as the official winner, and prints its stats.
"""
import os
import pickle
import neat

CHECKPOINT_DIR = os.path.expanduser(
    "~/Desktop/neat_bipedal_project/checkpoints"
)
RESULTS_DIR = os.path.expanduser(
    "~/Desktop/neat_bipedal_project/results"
)
CONFIG_PATH = os.path.expanduser(
    "~/Desktop/neat_bipedal_project/config-neat.txt"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load config
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    CONFIG_PATH
)

# Find all checkpoint files and sort numerically
checkpoint_files = sorted(
    [f for f in os.listdir(CHECKPOINT_DIR)
     if f.startswith("neat-checkpoint-")],
    key=lambda x: int(x.split("-")[-1])
)

print(f"Found {len(checkpoint_files)} checkpoints to scan")
print("Scanning for best genome...\n")

best_genome = None
best_fitness = float("-inf")
best_checkpoint = None

for fname in checkpoint_files:
    fpath = os.path.join(CHECKPOINT_DIR, fname)
    try:
        pop = neat.Checkpointer.restore_checkpoint(fpath)
        for genome_id, genome in pop.population.items():
            if genome.fitness is not None and genome.fitness > best_fitness:
                best_fitness = genome.fitness
                best_genome = genome
                best_checkpoint = fname
    except Exception as e:
        print(f"  Skipping {fname}: {e}")
        continue

    # Print progress every 100 checkpoints
    gen_num = int(fname.split("-")[-1])
    if gen_num % 100 == 0:
        print(f"  Scanned up to generation {gen_num} "
              f"| Best so far: {best_fitness:.2f}")

print(f"\n{'='*50}")
print(f"BEST GENOME FOUND")
print(f"{'='*50}")
print(f"  Fitness:     {best_fitness:.4f}")
print(f"  Nodes:       {len(best_genome.nodes)}")
print(f"  Connections: {len(best_genome.connections)}")
print(f"  Found in:    {best_checkpoint}")

# Save as official winner
winner_path = os.path.join(RESULTS_DIR, "winner_genome.pkl")
with open(winner_path, "wb") as f:
    pickle.dump(best_genome, f)
print(f"\n✓ Winner saved to {winner_path}")
print("Now run: python neat_bipedal.py --mode watch")
