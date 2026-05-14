#!/bin/bash -l
#SBATCH --job-name=setup_env
#SBATCH --partition=cpuq      # Change this if your cluster has a different standard CPU queue (e.g., 'standard' or 'interactive')
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G             # Plenty of memory for Conda to solve environments
#SBATCH --time=04:00:00       # Gives plenty of time for downloading Llama3:70b
#SBATCH --output=build_%j.out
#SBATCH --error=build_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "========================================"
echo "Starting build process on node: $HOSTNAME"
echo "Date: $(date)"
echo "========================================"

# Run the build script we created earlier
bash build.sh

echo "========================================"
echo "Build process finished at: $(date)"
echo "========================================"