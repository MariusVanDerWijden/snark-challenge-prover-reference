#!/bin/bash
mkdir build
pushd build
  cmake -DCMAKE_CXX_COMPILER=/usr/bin/g++-7 -DMULTICORE=ON -DUSE_PT_COMPRESSION=OFF .. 
  make -j12 main generate_parameters cuda_prover_piecewise
popd
mv build/libsnark/main
mv build/libsnark/generate_parameters .
mv build/cuda_prover_piecewise .
