import os
import pickle
import neat

CHECKPOINT_DIR = os.path.expanduser(
    "~/Desktop/neat_bipedal_project/checkpoints_ablation"
)
RESULTS_DIR = os.path.expanduser(
    "~/Desktop/neat_bipedal_project/results_ablation"
)
CONFIG_PATH = os.path.expanduser(
    "~/Desktop/neat_bipedal_project/config-neat-ablation.txt"
)

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    CONFIG_PATH
)

checkpoint_files = sorted(
    [f for f in os.listdir(CHECKPOINT_DIR)
     if f.startswith("ablation-checkpoint-")],
    key=lambda x: int(x.split("-")[-1])
)

print(f"Scanning {len(checkpoint_files)} checkpoints...")

best_genome  = None
best_fitness = float("-inf")
best_checkpoint = None

for fname in checkpoint_files:
    fpath = os.path.join(CHECKPOINT_DIR, fname)
    try:
        pop = neat.Checkpointer.restore_checkpoint(fpath)
        for genome_id, genome in pop.population.items():
            if genome.fitness is not None and genome.fitness > best_fitness:
                best_fitness    = genome.fitness
                best_genome     = genome
                best_checkpoint = fname
    except Exception as e:
        print(f"  Skipping {fname}: {e}")

print(f"\nBest ablation genome:")
print(f"  Fitness:     {best_fitness:.4f}")
print(f"  Nodes:       {len(best_genome.nodes)}")
print(f"  Connections: {len(best_genome.connections)}")
print(f"  Found in:    {best_checkpoint}")

with open(os.path.join(RESULTS_DIR, "winner_ablation.pkl"), "wb") as f:
    pickle.dump(best_genome, f)
print(f"\n✓ Saved to results_ablation/winner_ablation.pkl")
