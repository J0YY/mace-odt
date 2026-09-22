#!/usr/bin/env bash
#SBATCH --job-name=mace-odt-e10-pullback
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

project_dir=${MACE_ODT_PROJECT:-"/work/$USER/mace-odt"}
env_dir=${MACE_ODT_ENV:-"/work/$USER/envs/mace-odt-cpu"}
cache_dir=${MACE_ODT_CACHE:-"/work/$USER/cache/mace-odt"}
tmp_dir=${MACE_ODT_TMP:-"/work/$USER/tmp/mace-odt"}
dataset_dir=${MACE_ODT_DATASET:-"/work/$USER/datasets/mace-off23"}

mkdir -p "$cache_dir" "$tmp_dir"
export XDG_CACHE_HOME="$cache_dir"
export TMPDIR="$tmp_dir"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

cd "$project_dir"
mkdir -p logs results

"$env_dir/bin/python" -m mace_odt.cli.multi_consumer_fidelity \
  --xyz "$dataset_dir/test_large_neut_all.xyz" \
  --manifest results/mace_off23_test_manifest_64.json \
  --discovery-manifest results/mace_off23_discovery_manifest_128.json \
  --radial-npz results/mace_off23_small_radial_interface.npz \
  --multi-consumer-npz results/mace_off23_small_multi_consumer_environment_128.npz \
  --multi-consumer-json results/mace_off23_small_multi_consumer_environment_128.json \
  --first-branch-environment-npz results/mace_off23_small_global_environment_balanced.npz \
  --first-branch-environment-json results/mace_off23_small_global_environment_balanced.json \
  --first-branch-npz results/mace_off23_small_first_branch.npz \
  --model small \
  --device cpu \
  --dtype float64 \
  --ranks 48 64 80 96 \
  --random-seeds 20260924 20260925 20260926 20260927 20260928 \
  --output results/mace_off23_small_multi_consumer_fidelity_64.json
