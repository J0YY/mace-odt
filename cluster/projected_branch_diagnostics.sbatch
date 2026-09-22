#!/usr/bin/env bash
#SBATCH --job-name=mace-odt-e6-project
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

project_dir=${MACE_ODT_PROJECT:-"/work/$USER/mace-odt"}
env_dir=${MACE_ODT_ENV:-"/work/$USER/envs/mace-odt-cpu"}
cache_dir=${MACE_ODT_CACHE:-"/work/$USER/cache/mace-odt"}
tmp_dir=${MACE_ODT_TMP:-"/work/$USER/tmp/mace-odt"}

mkdir -p "$cache_dir" "$tmp_dir"
export XDG_CACHE_HOME="$cache_dir"
export TMPDIR="$tmp_dir"
export OMP_NUM_THREADS=8

cd "$project_dir"
mkdir -p logs results

"$env_dir/bin/python" -m mace_odt.cli.projected_branch_diagnostics \
  --radial-npz results/mace_off23_small_radial_interface.npz \
  --environment-npz results/mace_off23_small_global_environment_balanced.npz \
  --model small \
  --device cpu \
  --dtype float64 \
  --ranks 4 8 16 32 48 64 80 96 \
  --output results/mace_off23_small_projected_branch_balanced.json
