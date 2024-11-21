#!/bin/zsh

# Run ablation study with stage-based experiments

# Check if the stage parameter is set via CLI
if [ -z "$1" ]; then
  echo "Usage: run_experiments.sh <stage>"
  echo "  stage: The stage of the ablation study to run (1-4)"
  exit 1
fi

# Assign the CLI argument to a variable
STAGE=$1

# Check that exactly one stage parameter is set via CLI
if [ $(echo "$STAGE" | grep -o '[0-9]' | wc -l) -ne 1 ]; then
  echo "Error: The stage parameter must be a single digit."
  exit 1
fi

# Validate the value of stage
if [[ ! "$STAGE" =~ ^[1-4]$ ]]; then
  echo "Error: The stage parameter must be in the range 1 to 4."
  exit 1
fi

# Run the experiments based on the stage parameter
case $STAGE in
  1)
    echo "Running stage 1 - Baseline experiments without augmentation"
    uv run src/train.py --epochs 300 --num_filters 32 --depth 4
    uv run src/train.py --epochs 300 --num_filters 32 --depth 6
    uv run src/train.py --epochs 300 --num_filters 64 --depth 4
    ;;
  2)
    echo "Running stage 2 - Experiments with affine flip augmentation"
    uv run src/train.py --epochs 300 --num_filters 32 --depth 4 --augmentation affine_flip
    uv run src/train.py --epochs 300 --num_filters 32 --depth 6 --augmentation affine_flip
    uv run src/train.py --epochs 300 --num_filters 64 --depth 4 --augmentation affine_flip
    ;;
  3)
    echo "Running stage 3 - Experiments with artefact augmentation"
    uv run src/train.py --epochs 300 --num_filters 32 --depth 4 --augmentation artefact
    uv run src/train.py --epochs 300 --num_filters 32 --depth 6 --augmentation artefact
    uv run src/train.py --epochs 300 --num_filters 64 --depth 4 --augmentation artefact
    ;;
  4)
    echo "Running stage 4 - Experiments with affine flip and artefact augmentations"
    uv run src/train.py --epochs 300 --num_filters 32 --depth 4 --augmentation affine_flip artefact
    uv run src/train.py --epochs 300 --num_filters 32 --depth 6 --augmentation affine_flip artefact
    uv run src/train.py --epochs 300 --num_filters 64 --depth 4 --augmentation affine_flip artefact
    ;;
esac
