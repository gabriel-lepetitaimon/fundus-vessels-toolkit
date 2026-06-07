#!/bin/bash
    
#SBATCH --job-name={EXP}
#SBATCH --output={DIR}/out_{EXP}-%A-%a.out
#SBATCH --array=1,2,3,4
    
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=44:00:00
#SBATCH --mem=44G
#SBATCH --partition=liv4d
#SBATCH --gres=gpu:rtx2080ti:1

cd "/store-liv4d/travail/GNN-Fundus-galep/fundus-vessels-toolkit/train/Topo-GNN"
/store-liv4d/travail/GNN-Fundus-galep/miniconda3/bin/conda init
conda activate gnn
ulimit -n 2048

sleep $(( RANDOM % 11 )) # Sleep for a random time between 0 and 10 seconds to avoid starting all jobs at the same time

local retry_count=0
while true; do
    python exp_cli.py single-run {EXP_FILE}

    status=$?
    if [ $status -eq 20 ]; then
        echo "No remaining trials to run for experiment {EXP}."
        break
    elif [ $status -ne 0 ]; then
        echo "Experiment run failed with status $status. Retrying... (Attempt $((retry_count + 1)))"
        retry_count=$((retry_count + 1))
        if [ $retry_count -ge 3 ]; then
            echo "Experiment run failed after 3 attempts. Exiting."
            exit 1
        fi
        sleep 1  # Wait for 1 seconds before retrying
    fi
done
