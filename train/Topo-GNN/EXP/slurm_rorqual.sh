#!/bin/bash
    
#SBATCH --job-name={EXP}
#SBATCH --output={DIR}/out_{EXP}-%A-%a.out
#SBATCH --array=1:{N_RUNS}
    
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --mem=31G
#SBATCH --partition=liv4d
#SBATCH --gpus=h100_2g.20gb:1

module load python/3.13.2
module load scipy-stack/2026a
module load gcc opencv/4.13.0

cd "/home/galep/links/projects/def-fcheriet/galep/GNN-Topo/"
source ENV/bin/activate
export PYTHONPATH="${PYTHONPATH}:/home/galep/links/projects/def-fcheriet/galep/GNN-Topo/fundus-vessels-toolkit/src"

cd "fundus-vessels-toolkit/train/Topo-GNN"
ulimit -n 2048

python exp_cli.py single-run {EXP_FILE}

status=$?
if [ $status -eq 20 ]; then
    echo "No remaining trials to run for experiment {EXP}."
elif [ $status -ne 0 ]; then
    echo "Experiment run failed with status $status."
    exit 1
fi
