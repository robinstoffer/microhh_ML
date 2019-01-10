#!/bin/bash

# Loading modules
module load Python/3.6.1-intel-2016b
module load netCDF/4.4.1.1-intel-2016b
module load netCDF-C++4/4.3.0-intel-2016b

echo "Loaded modules are:"
module list

# Define virtualenv
#VIRTENV=tensorflow_cpu
#VIRTENV_ROOT=~/virtualenvs

VIRTENV=firstCNN_intel_CPU
VIRTENV_ROOT=~/virtualenv

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
export MPICC=mpicc
export MPICXX=mpicxx

# Tensorflow
echo "Installing Tensorflow"
pip install intel-tensorflow --no-cache-dir

# Getting this error when installing horovod, unless we comment a line: https://github.com/protocolbuffers/protobuf/issues/4069
sed -i 's/static_assert(std::is_pod<AuxillaryParseTableField>::value, "");/\/\/static_assert(std::is_pod<AuxillaryParseTableField>::value, "");/g' $VIRTENV_ROOT/$VIRTENV/lib/python3.6/site-packages/tensorflow/include/google/protobuf/generated_message_table_driven.h

# Horovod
echo "Installing Horovod"
pip install horovod --no-cache-dir

# netCDF
echo "Installing netCDF for python"
pip3 install netCDF4 --no-cache-dir

# Scipy
echo "Installing scipy for python"
pip3 install scipy --no-cache-dir

# Matplotlib
# Note: Tkinter is not supported with this installation
# Switch backend to Agg after importing matplotlib in your python session, then plot
echo "Installing matplotlib"
pip3 install matplotlib --no-cache-dir
