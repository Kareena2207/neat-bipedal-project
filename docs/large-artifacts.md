# Large Artifacts

This project originally contained large generated files, including training checkpoints, local virtual environments, and video recordings.

These are intentionally excluded from normal Git tracking:

- `neat_env/` local Python virtual environment
- `checkpoints/` and `checkpoints_*/` generated NEAT checkpoint histories
- `*.mov`, `*.mp4`, and `videos for yt/` demo videos
- `*.zip` generated archives

Recommended sharing approach:

1. Keep source code, configs, reports, plots, and small saved winner genomes in GitHub.
2. Upload large videos or full checkpoint folders to GitHub Releases or external storage.
3. Link those assets from the `README.md` if they are needed for a viewer or evaluator.
