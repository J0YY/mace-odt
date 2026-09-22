#!/usr/bin/env bash
#SBATCH --job-name=mace-odt-e9-discovery
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

project_dir=${MACE_ODT_PROJECT:-"/work/$USER/mace-odt"}
env_dir=${MACE_ODT_ENV:-"/work/$USER/envs/mace-odt-cpu"}
dataset_dir=${MACE_ODT_DATASET:-"/work/$USER/datasets/mace-off23"}

cd "$project_dir"
mkdir -p logs results

"$env_dir/bin/python" -m mace_odt.cli.discovery_manifest \
  --xyz "$dataset_dir/test_large_neut_all.xyz" \
  --evaluation-manifest results/mace_off23_test_manifest_64.json \
  --sample-size 128 \
  --seed 20260927 \
  --output results/mace_off23_discovery_manifest_128.json
