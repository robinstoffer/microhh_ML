#!/bin/bash

# Loading modules
module load Python/3.6.1-intel-2016b

echo "Loaded modules are:"
module list

# Define virtualenv
VIRTENV=tensorflow_cpu
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

# Tensorflow
echo "Installing Tensorflow"
pip install intel-tensorflow --no-cache-dir

# Horovod
echo "Installing Horovod"
pip install horovod --no-cache-dir

