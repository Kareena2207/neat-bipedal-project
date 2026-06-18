# NEAT Bipedal Walker

This project uses NEAT (NeuroEvolution of Augmenting Topologies) to evolve neural-network controllers for the `BipedalWalker-v3` environment from Gymnasium. It includes the baseline walker, multi-objective experiments, hardcore terrain experiments, curriculum learning, ablation analysis, visualizations, and saved winner genomes.

## Project Structure

```text
.
├── neat_bipedal.py                  # Main baseline NEAT training, watch, and visualization script
├── config-neat.txt                  # NEAT configuration for the baseline experiment
├── config-neat-ablation.txt         # NEAT configuration for ablation experiments
├── experiment2_multiobjective.py    # Multi-objective experiment
├── experiment3_hardcore.py          # Hardcore terrain experiment
├── experiment4_curriculum.py        # Curriculum learning experiment
├── ablation_study.py                # Ablation experiment
├── evaluate_all.py                  # Evaluation helper for saved winners
├── plot_all_experiments.py          # Combined result plotting
├── watch_*.py                       # Scripts for watching trained agents
├── extract_winner.py                # Extracts winner genomes from checkpoints
├── extract_ablation_winner.py       # Extracts ablation winner genomes
├── results*/                        # Lightweight plots and saved winner genomes
├── Final_presentation_bio_ai.pdf    # Final presentation PDF
├── Final_presentation_group11_bio_ai.pptx
├── extended_abstract_ieee.tex
└── docs/
    └── large-artifacts.md
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, if Box2D installation fails, install system build tools first:

```bash
xcode-select --install
```

## Run the Baseline Experiment

Train from scratch:

```bash
python neat_bipedal.py --mode train
```

Watch the saved winner genome:

```bash
python neat_bipedal.py --mode watch
```

Generate visualizations:

```bash
python neat_bipedal.py --mode visualize
```

Resume from a checkpoint:

```bash
python neat_bipedal.py --mode checkpoint
```

## Experiments

The project includes several experiment scripts:

```bash
python experiment2_multiobjective.py
python experiment3_hardcore.py
python experiment4_curriculum.py
python ablation_study.py
python evaluate_all.py
python plot_all_experiments.py
```

The `watch_*.py` scripts render trained agents for different experiment variants.

## GitHub Upload Notes

The repository is prepared so source code, configs, reports, plots, and winner genomes can be tracked in Git. Generated checkpoints, local virtual environments, archives, and large video files are ignored by `.gitignore` because they make the repository too large and slow to clone.

If you want to share demo videos or the full checkpoint history, upload them separately through GitHub Releases, Google Drive, or another storage link, then add the links here.

## License

Add a license before making the repository public if you want others to reuse the code. MIT is a common choice for academic/demo projects.
