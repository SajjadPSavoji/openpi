#!/bin/bash -l
#=========================================================================================
# Slurm SBATCH Directives
#=========================================================================================
#SBATCH --job-name=openpi
#SBATCH --partition=compute                
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 
#SBATCH --time=7-01:00:00 
#SBATCH --output=/shared_work/sajjad/slurm-logs/%x-%j.out
#SBATCH --error=/shared_work/sajjad/slurm-logs/%x-%j.err
#SBATCH --cpus-per-task=5
#SBATCH --mem=200G 
#=========================================================================================

set -euo pipefail

echo "Job started on $(hostname) at $(date)"

# Clear existing module state
module purge || true

# ---- Setup UV ----
module load uv

# Explicitly set your personal paths to avoid "Permission Denied" on /pm/uv
export UV_CACHE_DIR="/pm/sajjad/.cache/uv"
export UV_PYTHON_INSTALL_DIR="/pm/sajjad/uv/python"
export UV_PYTHON_BIN_DIR="/pm/sajjad/uv/bin"
export UV_PYTHON_CACHE_DIR="/pm/sajjad/uv/python-cache"
export UV_TOOL_DIR="/pm/sajjad/uv/tools"
export HF_HOME="/shared_work/sajjad/.cache/huggingface"

# ---- Activate env ----
source /pm/sajjad/openpi/bin/activate

# ---- Verification (Optional but helpful for logs) ----
echo "Using Python: $(which python)"
echo "UV Cache: $UV_CACHE_DIR"

# ---- Your workload ----
nvidia-smi || true

# Run the motion planning script
CUDA_VISIBLE_DEVICES=0 uv run --active scripts/compute_norm_stats.py --config-name pi0_anybox

echo "Job finished at $(date)"