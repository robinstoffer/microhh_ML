#!/bin/bash

# Loading modules
module load Python/3.6.1-intel-2016b

module load CUDA/9.0.176
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.2.12-CUDA-9.0.176

echo "Loaded modules are:"
module list

# Setting ENV variables
export HOROVOD_CUDA_HOME=$CUDA_HOME
export HOROVOD_CUDA_INCLUDE=$CUDA_HOME/include
export HOROVOD_CUDA_LIB=$CUDA_HOME/lib64
export HOROVOD_NCCL_HOME=$EBROOTNCCL
export HOROVOD_GPU_ALLREDUCE=NCCL

# Define virtualenv
VIRTENV=tensorflow_gpu
VIRTENV_ROOT=~/virtualenvs

# Creating virtual env
echo "Creating virtual environment $VIRTENV_ROOT/$VIRTENV"
virtualenv $VIRTENV_ROOT/$VIRTENV

# Sourcing virtual env
echo "Sourching virtual environment $VIRTENV_ROOT/$VIRTENV"
source $VIRTENV_ROOT/$VIRTENV/bin/activate

# Check current python packages
echo "Current Python packages"
pip list

# Export LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SURFSARA_LIBRARY_PATH:$LD_LIBRARY_PATH
export MPICC=mpicc
export MPICXX=mpicxx
export HOROVOD_MPICXX_SHOW="mpicxx --showme:link"

# Tensorflow
echo "Installing Tensorflow"
pip3 install tensorflow-gpu --no-cache-dir

# Horovod
echo "Installing Horovod"
pip3 install horovod --no-cache-dir

