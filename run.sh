#!/bin/bash
#SBATCH --job-name=run
#SBATCH --account=lcn 	#	lcn lcnrtx
#SBATCH --partition=rtx6000		#lcnv100 rtx6000    # Specify the partition name (e.g., main, gpu, etc.)
#SBATCH --nodes=1                 # Number of nodes (1 for a single node)
#SBATCH --ntasks-per-node=1       # Number of tasks per node (1 for a single task)
#SBATCH --cpus-per-task=3         # Number of CPU cores per task (adjust as needed)
#SBATCH --gres=gpu:1    # Request GPU resources if needed
#SBATCH --gpus=1
#SBATCH --time=7-00:00:00       # Requested time limit (e.g., 1:00:00 for 1 hour)
#SBATCH --mem=20G      # Requested memory per node (e.g., 4G for 4 gigabytes)
#SBATCH --output=run.log

# Load necessary modules (adjust as needed)
#imodule load anaconda/3.7  # Load Anaconda module

# Activate a conda environment if needed
source ~/.bashrc

source activate py3.8

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/autofs/space/bal_004/users/jd1677/anaconda3/envs/py3.9/lib
# Launch JupyterLab

#python lasso.py --gameweek 10



for ((gameweek=16; gameweek<=18; gameweek++))
do
    echo "Running for gameweek $gameweek:"
    python lgbm.py --gameweek $gameweek
    python lasso.py --gameweek $gameweek
    python cnn.py --gameweek $gameweek
    python lstm.py --gameweek $gameweek
done

