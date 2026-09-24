#!/usr/bin/env bash
#SBATCH --job-name=mace-odt-t4-exact-fidelity
#SBATCH --partition=compute
#SBATCH --time=12:00:00
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

require_sha256() {
  local path=$1
  local expected=$2
  test -s "$path"
  echo "$expected  $path" | sha256sum --check --status
}

require_sha256 results/mace_off23_test_manifest_64.json 5e7314c41fedbb614a457d7f4b91c87e615f08f95a35e843b224cd0b9aa82835
require_sha256 results/mace_off23_discovery_manifest_128.json a624def23b363155da28a148998802438ab756f7c698161bc388ebed230d653c
require_sha256 results/mace_off23_small_radial_interface.npz 564776ad8169c5f26cee0335adbe497f64facbee5073dabf5341109aeff8be86
require_sha256 results/mace_off23_small_global_environment_balanced.json 082deae6e5f1d1416c15d0c28c201348ed208a7773b93965bee89886c86267f8
require_sha256 results/mace_off23_small_global_environment_balanced.npz 8f7a2225580d60b526554885f7a80e3339b19027a7537e51eae95932f08219a9
require_sha256 results/mace_off23_small_multi_consumer_environment_128.json 26c1ca405ccfa09b66438c301548dd686e36ca897c6fd62450b27d9fab89f46f
require_sha256 results/mace_off23_small_multi_consumer_environment_128.npz b2e7cb6457a8584bc754a1fe1301d719f800b26b2fd5676bb17383db314badb2
require_sha256 configs/immediate_consumer_fidelity_v1.json 5e2bb8cd5dd1ea88e53bca94fe4864bea19c81aa64559019a8fd3331343ad775

test -s results/mace_off23_small_immediate_consumer_environment.json
test -s results/mace_off23_small_immediate_consumer_environment.npz
test ! -e results/mace_off23_small_immediate_consumer_fidelity_64.json

"$env_dir/bin/python" -m mace_odt.cli.immediate_consumer_fidelity \
  --xyz "$dataset_dir/test_large_neut_all.xyz" \
  --manifest results/mace_off23_test_manifest_64.json \
  --discovery-manifest results/mace_off23_discovery_manifest_128.json \
  --radial-npz results/mace_off23_small_radial_interface.npz \
  --environment-npz results/mace_off23_small_immediate_consumer_environment.npz \
  --environment-json results/mace_off23_small_immediate_consumer_environment.json \
  --first-branch-environment-npz results/mace_off23_small_global_environment_balanced.npz \
  --first-branch-environment-json results/mace_off23_small_global_environment_balanced.json \
  --multi-consumer-npz results/mace_off23_small_multi_consumer_environment_128.npz \
  --multi-consumer-json results/mace_off23_small_multi_consumer_environment_128.json \
  --protocol-config configs/immediate_consumer_fidelity_v1.json \
  --model small \
  --device cpu \
  --dtype float64 \
  --ranks 16 32 48 64 80 96 \
  --random-seeds 20260924 20260925 20260926 20260927 20260928 \
  --output results/mace_off23_small_immediate_consumer_fidelity_64.json
