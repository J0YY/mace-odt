#!/usr/bin/env bash
#SBATCH --job-name=mace-odt-e5-global
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

project_dir=${MACE_ODT_PROJECT:-"/work/$USER/mace-odt"}
env_dir=${MACE_ODT_ENV:-"/work/$USER/envs/mace-odt-cpu"}
tmp_dir=${MACE_ODT_TMP:-"/work/$USER/tmp/mace-odt"}

mkdir -p "$tmp_dir"
export TMPDIR="$tmp_dir"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

cd "$project_dir"
mkdir -p logs results

"$env_dir/bin/python" -m mace_odt.cli.global_environment_audit \
  --radial-npz results/mace_off23_small_radial_interface.npz \
  --branch-npz results/mace_off23_small_first_branch.npz \
  --output results/mace_off23_small_global_environment.json \
  --npz results/mace_off23_small_global_environment.npz \
  --uniform-ranks 0 4 8 16 32 48 64 80 96
