#!/bin/bash

#SBATCH --chdir=./                       # Set the working directory
#SBATCH --job-name=embed_col_docs              # Name to show in the job queue
#SBATCH --output=./output/job.%j.out     # Name of stdout output file (%j expands to jobId)
#SBATCH --ntasks=1                      # Total number of mpi tasks requested (CPU Cores basically)
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1                        # Total number of nodes requested
#SBATCH --nodelist=                          # Target node (cluster-specific, configure before use)
#SBATCH --partition=gpu               # Partition (a.k.a. queue) to use
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00                # Max run time (days-hh:mm:ss) ... adjust as necessary

uv run setup_col_embeddings.py
