#!/bin/bash

#SBATCH --chdir=./                       # Set the working directory
#SBATCH --mail-user=EMAIL_REDACTED    # Who to send emails to
#SBATCH --mail-type=ALL                  # Send emails on start, end and failure
#SBATCH --job-name=embed_pages              # Name to show in the job queue
#SBATCH --output=./output/job.%j.out     # Name of stdout output file (%j expands to jobId)
#SBATCH --ntasks=1                      # Total number of mpi tasks requested (CPU Cores basically)
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1                        # Total number of nodes requested
#SBATCH --exclude=GPU_NODE_RANGE_REDACTED
#SBATCH --nodelist=GPU_NODE_REDACTED
#SBATCH --partition=gpu               # Partition (a.k.a. queue) to use
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00                # Max run time (days-hh:mm:ss) ... adjust as necessary

uv run setup.py
