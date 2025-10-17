#!/bin/sh

#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 24:00:00
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --tasks-per-node 1
#SBATCH --job-name turbulent_radiative_layer_2D

module purge
module load baskerville
module load bask-apps/live
module load Python/3.10.8-GCCcore-12.2.0
module load FFmpeg/5.1.2-GCCcore-12.2.0

BASE_PATH=/bask/homes/l/ltcx7228/vjgo8416-ai-phy-sys/

source .venv/bin/activate

RUN_PATH=autoemulate/experimental


## To fix this error:
## RuntimeError: Deterministic behavior was enabled with either
## torch.use_deterministic_algorithms(True) or at::Context::setDeterministicAlgorithms(true),
## but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2.
## To enable deterministic behavior in this case, you must set an environment variable before
## running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8.
## For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python $RUN_PATH/run_the_well_experiment.py --config $RUN_PATH/configs/turbulent_radiative_layer_2d.yaml
