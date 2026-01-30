# Slurm templates (cs336dev)

These scripts are **templates**. You likely need to adjust:
- `--partition` (and maybe `--qos`)
- time limits
- CPU/memory per task
- `--gres=gpu:1` vs `--gpus=1` depending on your Slurm setup

Typical workflow:

1) Inspect cluster resources
- `sinfo -o "%P %a %l %D %G"`
- `scontrol show partition <partition>`

2) Submit jobs
- `sbatch scripts/slurm/pretokenize_tinystories.sbatch`
- `sbatch scripts/slurm/train_tinystories.sbatch`
- `sbatch scripts/slurm/lr_sweep_array.sbatch`

3) Monitor
- `squeue -u $USER`
- `tail -f slurm_logs/<jobname>_<jobid>.out`
- `scancel <jobid>`
