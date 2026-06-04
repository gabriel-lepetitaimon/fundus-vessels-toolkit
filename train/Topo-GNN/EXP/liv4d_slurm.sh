#!/bin/bash
    
#SBATCH --job-name={EXP}
#SBATCH --output=out_{EXP}.out
#SBATCH --array=0-4
    
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --partition=liv4d
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --exclusive

cd "/store-liv4d/travail/GNN-Fundus-galep/fundus-vessels-toolkit/train/Topo-GNN"
/store-liv4d/travail/GNN-Fundus-galep/miniconda3/bin/conda init
conda activate gnn

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
