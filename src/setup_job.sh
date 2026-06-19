#!/bin/bash

#SBATCH --chdir=./                       # Set the working directory
#SBATCH --mail-user=EMAIL_REDACTED    # Who to send emails to
#SBATCH --mail-type=ALL                  # Send emails on start, end and failure
#SBATCH --job-name=setup_db              # Name to show in the job queue
#SBATCH --output=./output/job.%j.out     # Name of stdout output file (%j expands to jobId)
#SBATCH --ntasks=16                      # Total number of mpi tasks requested (CPU Cores basically)
#SBATCH --nodes=1                        # Total number of nodes requested
#SBATCH --partition=normal               # Partition (a.k.a. queue) to use
#SBATCH --time=0-12:00:00                # Max run time (days-hh:mm:ss) ... adjust as necessary

uv run setup.py
