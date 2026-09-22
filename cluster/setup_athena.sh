#!/usr/bin/env bash
set -euo pipefail

project_dir=${1:-"/work/$USER/mace-odt"}
env_dir=${2:-"/work/$USER/envs/mace-odt-cpu"}
cache_dir=${MACE_ODT_CACHE:-"/work/$USER/cache/mace-odt"}
tmp_dir=${MACE_ODT_TMP:-"/work/$USER/tmp/mace-odt"}
uv_bin=${MACE_ODT_UV:-"$HOME/.local/bin/uv"}

mkdir -p "$cache_dir" "$tmp_dir"
export XDG_CACHE_HOME="$cache_dir"
export PIP_CACHE_DIR="$cache_dir/pip"
export TMPDIR="$tmp_dir"
export UV_CACHE_DIR="$cache_dir/uv"
export UV_PYTHON_INSTALL_DIR="/work/$USER/python"

if [[ ! -x "$env_dir/bin/python" ]] || ! "$env_dir/bin/python" -m pip --version >/dev/null 2>&1; then
  "$uv_bin" venv --clear --seed --python 3.10 "$env_dir"
fi
"$env_dir/bin/python" -m pip install --upgrade pip setuptools wheel
"$env_dir/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.6.0
"$env_dir/bin/python" -m pip install -e "$project_dir[dev]"

"$env_dir/bin/python" -m mace_odt.cli.environment_audit \
  --output "$project_dir/results/athena_environment.json"

printf 'environment: %s\n' "$env_dir"
printf 'project: %s\n' "$project_dir"
printf 'cache: %s\n' "$cache_dir"
