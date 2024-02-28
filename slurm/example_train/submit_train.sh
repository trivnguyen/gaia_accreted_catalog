#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -A PHY210068P
#SBATCH --gres=gpu:v100-16:1
#SBATCH --nodes 1
#SBATCH --job-name train_0
#SBATCH --time 10:00:00
#SBATCH --mail-type all
#SBATCH --mail-user tnguy@mit.edu
#SBATCH -o out.out
#SBATCH -e err.err

# activate conda environment
if [ -f "/ocean/projects/ast200012p/tvnguyen/miniconda3/etc/profile.d/conda.sh" ]; then
    . "/ocean/projects/ast200012p/tvnguyen/miniconda3/etc/profile.d/conda.sh"
else
    export PATH="/ocean/projects/ast200012p/tvnguyen/miniconda3/bin:$PATH"
fi
conda activate jeans-gnn-new

config=$(realpath config.py)
cd /jet/home/tvnguyen/accreted_catalog/gaia_accreted_catalog
python train.py --config $config

exit 0
