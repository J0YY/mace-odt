#!/usr/bin/env bash
#SBATCH --job-name=mace-odt-e4-compile
#SBATCH --partition=compute
#SBATCH --time=01:00:00
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
export PIP_CACHE_DIR="$cache_dir/pip"
export TMPDIR="$tmp_dir"
export OMP_NUM_THREADS=8

cd "$project_dir"
mkdir -p logs results

"$env_dir/bin/python" -m mace_odt.cli.first_branch_compile \
  --model small \
  --device cpu \
  --dtype float64 \
  --output results/mace_off23_small_first_branch.json \
  --npz results/mace_off23_small_first_branch.npz
