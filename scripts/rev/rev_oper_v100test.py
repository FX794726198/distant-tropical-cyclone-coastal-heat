"""rev13: can the 0.25-degree model run on a 32 GB V100?  One hot case, one lead, CTL only, through the same code path as
rev_run.py --model oper (rev_build patches, gc_run.main).  Writes to $TCH/out_oper_v100test, not to out_oper_rev.
Prints wall time, JAX peak GPU memory and the compile times logged by JAX (to separate them from the forecast itself).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat')), 'scripts'))
import jax
jax.config.update('jax_log_compiles', True)
import rev_build  # noqa: F401
import gc_run

sys.argv = [sys.argv[0], '--model', 'oper', '--variants', 'CTL', '--leads', '48', '--cases', '6-6', '--tag', '_v100test']
t0 = time.time()
gc_run.main()
st = jax.devices()[0].memory_stats() or {}
print(f'V100TEST wall {time.time() - t0:.0f} s; peak_bytes_in_use {st.get("peak_bytes_in_use", 0) / 2**30:.2f} GiB; '
      f'bytes_limit {st.get("bytes_limit", 0) / 2**30:.2f} GiB; device {jax.devices()[0].device_kind}', flush=True)
