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

environment_json=results/mace_off23_small_immediate_consumer_environment.json
environment_npz=results/mace_off23_small_immediate_consumer_environment.npz

run_environment() {
  local output_json=$1
  local output_npz=$2
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
    --output "$output_json" \
    --npz "$output_npz"
}

validate_pair() {
  "$env_dir/bin/python" - "$1" "$2" <<'PY'
import hashlib
import json
import pathlib
import sys

metadata = json.loads(pathlib.Path(sys.argv[1]).read_text())
digest = hashlib.sha256(pathlib.Path(sys.argv[2]).read_bytes()).hexdigest()
if not metadata.get("gate_passed") or metadata.get("npz_sha256") != digest:
    raise SystemExit("exact environment JSON and NPZ failed validation")
PY
}

if test -s "$environment_json" && test -s "$environment_npz"; then
  validate_pair "$environment_json" "$environment_npz"
elif test -s "$environment_json" && test ! -e "$environment_npz"; then
  regeneration_dir=$(mktemp -d "$tmp_dir/mace-odt-environment.XXXXXX")
  generated_json="$regeneration_dir/environment.json"
  generated_npz="$regeneration_dir/environment.npz"
  run_environment "$generated_json" "$generated_npz"
  validate_pair "$generated_json" "$generated_npz"
  "$env_dir/bin/python" - "$environment_json" "$generated_json" <<'PY'
import json
import pathlib
import sys

committed = json.loads(pathlib.Path(sys.argv[1]).read_text())
generated = json.loads(pathlib.Path(sys.argv[2]).read_text())
for name in ("npz_sha256", "method", "required_consumers", "exactness_tolerance"):
    if committed.get(name) != generated.get(name):
        raise SystemExit(f"regenerated environment differs in {name}")
PY
  mv "$generated_npz" "$environment_npz"
  rm "$generated_json"
  rmdir "$regeneration_dir"
elif test ! -e "$environment_json" && test ! -e "$environment_npz"; then
  run_environment "$environment_json" "$environment_npz"
else
  echo "refusing an incomplete exact environment pair" >&2
  exit 1
fi
