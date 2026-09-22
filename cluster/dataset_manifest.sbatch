#!/usr/bin/env bash
#SBATCH --job-name=mace-odt-e7-manifest
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

"$env_dir/bin/python" -m mace_odt.cli.dataset_manifest \
  --xyz "$dataset_dir/test_large_neut_all.xyz" \
  --sample-size 64 \
  --seed 20260922 \
  --archive-id cb8351dd-f09c-413f-921c-67a702a7f0c5 \
  --archive-md5 bf1f7c7ff1714e6ac6af2cdc642a130a \
  --source-url https://doi.org/10.17863/CAM.107498 \
  --output results/mace_off23_test_manifest_64.json
