#!/bin/bash

#Start interactive job on Haswell node that supports AVX2-set.
#srun -t 60 -N 1 --partition=short --constraint=haswell --pty bash -il

#Activate virtual environment
source ~/virtualenv/firstCNN_intel_CPU/bin/activate

#Go to directory with Neural Network
cd ~/microhh/cases/moser600/git_repository/Neural\ Network

#Load modules new environment
module purge
module load surf-devel
module load 2019
#module load intel/2018b
module load netCDF/4.6.1-intel-2018b
#module load netCDF-C++4/4.3.0-intel-2018b
#module load CMake/3.12.1-GCCcore-7.3.0
#module load cuDNN/7.6.3-CUDA-10.0.130
#module load FFTW/3.3.8-intel-2018b
#module load Doxygen/1.8.14-GCCcore-7.3.0
#module load OpenBLAS/0.3.1-GCC-7.3.0-2.30
module load imkl/2018.3.222-iimpi-2018b
module load Valgrind/3.14.0-intel-2018b

#Compile cpp inference scripts

#GCC compiler
#g++ -Wall -o MLP diff_U.h diff_U.cpp Grid.h Grid.cpp main_test.cpp Network.h Network.cpp -lnetcdf -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -std=c++14 -Ofast -march=native

#g++ -Wall -o MLP diff_U.h diff_U.cpp Grid.h Grid_test.cpp main_test.cpp Network.h Network.cpp -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -std=c++14 -Ofast -march=native

#Intel compiler (Intel MKL)
icpc -Wall -o MLP diff_U.h diff_U.cpp Grid.h Grid.cpp main.cpp Network.h Network.cpp -lnetcdf -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -std=c++14 -Ofast -xAVX -axCORE-AVX-I,CORE-AVX2,CORE-AVX512 -ipo

#icpc -Wall -o MLP diff_U.h diff_U.cpp Grid.h Grid_test.cpp main_test.cpp Network.h Network.cpp -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -std=c++14 -Ofast -xAVX -axCORE-AVX-I,CORE-AVX2,CORE-AVX512 -g

#cblas_sgemm_test
#icpc -Wall -o cblas_sgemm_test cblas_sgemm_test.cpp -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -std=c++14 -Ofast -xAVX -axCORE-AVX-I,CORE-AVX2,CORE-AVX512

#conv_test
#icpc -Wall -o conv_test conv_test.cpp -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -std=c++14 -Ofast -xAVX -axCORE-AVX-I,CORE-AVX2,CORE-AVX512 -ipo
#Intel compiler (openBLAS)
#icpc -Wall -o MLP diff_U.h diff_U.cpp Grid.h Grid.cpp main_test.cpp Network.h Network.cpp -lnetcdf -lopenblas -lpthread -lm -ldl -std=c++14 -Ofast -xAVX -axCORE-AVX-I,CORE-AVX2,CORE-AVX512

#Execute created executable
#valgrind --tool=callgrind ./MLP
./MLP
#./conv_test
#./cblas_sgemm_test
