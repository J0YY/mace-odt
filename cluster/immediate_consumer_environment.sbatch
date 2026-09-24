#!/usr/bin/env bash
#SBATCH --job-name=mace-odt-t3-exact-consumers
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
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

"$env_dir/bin/python" -m mace_odt.cli.immediate_consumer_environment \
  --xyz "$dataset_dir/test_large_neut_all.xyz" \
  --manifest results/mace_off23_discovery_manifest_128.json \
  --radial-npz results/mace_off23_small_radial_interface.npz \
  --model small \
  --device cpu \
  --dtype float64 \
  --ranks 16 32 48 64 80 96 \
  --order-weights 1 1 1 \
  --consumer-weights 1 1 1 \
  --exactness-tolerance 5e-9 \
  --expected-checkpoint-sha256 165cce4cfec5a34b9c64d4ebf95de15d71106bb584b7291c8470f0749977c46f \
  --expected-manifest-sha256 a624def23b363155da28a148998802438ab756f7c698161bc388ebed230d653c \
  --expected-radial-sha256 564776ad8169c5f26cee0335adbe497f64facbee5073dabf5341109aeff8be86 \
  --output results/mace_off23_small_immediate_consumer_environment.json \
  --npz results/mace_off23_small_immediate_consumer_environment.npz
