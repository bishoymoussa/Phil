#!/bin/bash
#SBATCH --job-name=phil-venv
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/galoaa.b/phillm/logs/bootstrap_venv_%j.out
#SBATCH --error=/scratch/galoaa.b/phillm/logs/bootstrap_venv_%j.err

set -euo pipefail
bash /scratch/galoaa.b/phillm/Phil/scripts/explorer/bootstrap_venv.sh
