#!/bin/bash
    
#SBATCH --job-name={EXP}
#SBATCH --output=tmp/slurm/out_{EXP}_%A-%a.out
#SBATCH --array=0-{N_RUNS}%4
    
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=90:00
#SBATCH --mem=32G
#SBATCH --partition=liv4d
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --exclusive

cd "/store-liv4d/travail/GNN-Fundus-galep/fundus-vessels-toolkit/train/Topo-GNN"
conda activate gnn
python exp_run.py single-run {EXP_FILE}
