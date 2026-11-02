"""gc_run.py with the revision variants: python rev_run.py --model small --variants CTL,WIDEL,W1000 --tag _rev"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat')), 'scripts'))
# rev13 split: a model's runs can be moved to their own job (rev_small.slurm, rev_oper_v100.slurm), which writes
# rev/<model>_split.claimed when it starts and runs with REV_<MODEL>_OWNER=1; any other caller then leaves that model
# alone so the two jobs never write the same files.
_m = sys.argv[sys.argv.index('--model') + 1] if '--model' in sys.argv else 'small'
if os.environ.get(f'REV_{_m.upper()}_OWNER') != '1' and \
        os.path.exists(os.path.join(os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat')), 'rev', f'{_m}_split.claimed')):
    print(f'rev_run: {_m} model belongs to the separate job (rev/{_m}_split.claimed), skipped', flush=True); sys.exit(0)
import rev_build  # noqa: F401  (patches gc_build.modify)
import gc_run
gc_run.main()
