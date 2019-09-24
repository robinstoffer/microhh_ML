#!/bin/bash

#Activate virtual environment
source ~/virtualenv/firstCNN_intel_CPU/bin/activate

#Go to directory with Neural Network
cd ~/microhh/cases/moser600/git_repository/Neural\ Network

#Load modules new environment
module purge
module load surf-devel
module load 2019
module load intel/2018b
module load netCDF/4.6.1-intel-2018b
module load netCDF-C++4/4.3.0-intel-2018b
module load CMake/3.12.1-GCCcore-7.3.0
module load cuDNN/7.6.3-CUDA-10.0.130
module load FFTW/3.3.8-intel-2018b
module load Doxygen/1.8.14-GCCcore-7.3.0
module load OpenBLAS/0.3.1-GCC-7.3.0-2.30

#Compile cpp inference scripts
g++ -Wall -o MLP diff_U.h diff_U.cpp Grid.h Grid.cpp main.cpp Network.h Network.cpp -lnetcdf -lopenblas -std=c++14 

#Execute created executable
./MLP
