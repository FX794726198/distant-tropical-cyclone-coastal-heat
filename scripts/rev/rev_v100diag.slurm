#!/bin/bash
#SBATCH --job-name=v100diag --partition=gpu-v100 --account=innovation --ntasks=1 --cpus-per-task=2 --gpus-per-task=1 --mem-per-cpu=4000M --time=00:10:00 --output=/scratch/%u/tcheat/logs/rev_v100diag_%j.log
# rev13: why does the gc venv die with "Illegal instruction" on gpu-v100? CPU model/flags and each import in its own process.
module load 2025 python/3.11.9 2>/dev/null
hostname; lscpu | grep -E "Model name|^Flags" | sed 's/Flags:/Flags:/' | cut -c1-2000
echo "python -> $(readlink -f ~/venvs/gc/bin/python)"; cat ~/venvs/gc/pyvenv.cfg
for m in sys numpy scipy pandas xarray netCDF4 jaxlib jax haiku graphcast; do
  ~/venvs/gc/bin/python -c "import $m" >/dev/null 2>&1; echo "import $m rc=$?"
done
~/venvs/gc/bin/python -c "import jax; print(jax.devices())" 2>&1 | tail -2; echo "jax.devices rc=$?"
~/venvs/gc/bin/python -c "import jax.numpy as jnp; print(float(jnp.ones(3).sum()))" 2>&1 | tail -1
