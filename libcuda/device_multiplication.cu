/*****************************************************************************
 Implementation of multiplication + exponentiation on Finite Elements
 *****************************************************************************
 * @author     Marius van der Wijden
 * Copyright [2019] [Marius van der Wijden]
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/
 
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <stdio.h>
#include <cassert>

#include "device_multiplication.h"

#include "device_field.h"
#include "device_field_operators.h"

#define GRID_SIZE 32
#define BLOCK_SIZE 16 

template<typename FieldT> 
__device__ __constant__ FieldT zero;
template<typename T>
__device__ T out;

template <typename T, unsigned int blockSize>
__device__ void warpReduce(T *tmpData, unsigned int tid) {
    if (blockSize >= 64) tmpData[tid] += tmpData[tid + 32];
    if (blockSize >= 32) tmpData[tid] += tmpData[tid + 16];
    if (blockSize >= 16) tmpData[tid] += tmpData[tid + 8];
    if (blockSize >= 8) tmpData[tid] += tmpData[tid + 4];
    if (blockSize >= 4) tmpData[tid] += tmpData[tid + 2];
    if (blockSize >= 2) tmpData[tid] += tmpData[tid + 1];
}

template <typename T, typename FieldT, unsigned int blockSize>
__global__ void device_multi_exp_inner(T *vec, FieldT *scalar, size_t field_size) {
    extern __shared__ T tmpData[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;

    tmpData[tid] = zero<T>;
    while (i < field_size) { 
        vec[i] *= scalar[i];
        tmpData[tid] += vec[i]; 
        i += gridSize; 
    }
    __syncthreads();
    
    if (blockSize >= 2048) { if (tid < 1024) { tmpData[tid] += tmpData[tid + 1024]; } __syncthreads(); }
    if (blockSize >= 1024) { if (tid < 512) { tmpData[tid] += tmpData[tid + 512]; } __syncthreads(); }
    if (blockSize >= 512) { if (tid < 256) { tmpData[tid] += tmpData[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { tmpData[tid] += tmpData[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { tmpData[tid] += tmpData[tid + 64]; } __syncthreads(); }

    if (tid < 32) warpReduce<T,blockSize>(tmpData, tid);
    if (tid == 0) out<T> = tmpData[0];
}

template<typename T, typename FieldT>
T cuda_multi_exp_inner(
    typename std::vector<T>::const_iterator vec_start,
    typename std::vector<T>::const_iterator vec_end,
    typename std::vector<FieldT>::const_iterator scalar_start,
    typename std::vector<FieldT>::const_iterator scalar_end)
{
    printf("Enter custom gpu code");
    T *d_vec;
    FieldT *d_scalar;
    size_t vec_size = (vec_end - vec_start) * sizeof(T);
    size_t scalar_size = (scalar_end - scalar_start) * sizeof(FieldT);

    cudaMalloc((void **)&d_vec, vec_size);
    cudaMalloc((void **)&d_scalar, scalar_size);
	
    cudaMemcpy(d_vec, &vec_start, vec_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scalar, &scalar_start, scalar_size, cudaMemcpyHostToDevice);
    uint smemSize = vec_size / 2;	

    dim3 dimGrid (GRID_SIZE, GRID_SIZE);
    dim3 dimBlock (BLOCK_SIZE, BLOCK_SIZE);
    uint threads = dimGrid.x * dimGrid.y;
    switch (threads)
    {
        case 1024:
        device_multi_exp_inner<T,FieldT,2048>
          <<< dimGrid, dimBlock, smemSize >>>(d_vec, d_scalar, vec_size); break;
        /*
        case 2048:
        device_multi_exp_inner<fields::Field,fields::Field,1024>
          <<< dimGrid, dimBlock, smemSize >>>(d_vec, d_scalar, vec_size); break;
        case 512:
        device_multi_exp_inner<fields::Field,fields::Field,512>
          <<< dimGrid, dimBlock, smemSize >>>(d_vec, d_scalar, vec_size); break;
        */
    }

    T* res; T result;
    cudaGetSymbolAddress((void**) &res, out<T>);
    cudaMemcpy(&result, res, sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_vec);
    cudaFree(d_scalar);
    return result;
}
/*
template int cuda_multi_exp_inner<int,uint>(
    typename std::vector<int>::const_iterator vec_start,
    typename std::vector<int>::const_iterator vec_end,
    typename std::vector<uint>::const_iterator scalar_start,
    typename std::vector<uint>::const_iterator scalar_end);*/

template fields::FieldElement cuda_multi_exp_inner<fields::FieldElement,fields::Scalar>(
    typename std::vector<fields::FieldElement>::const_iterator vec_start,
    typename std::vector<fields::FieldElement>::const_iterator vec_end,
    typename std::vector<fields::Scalar>::const_iterator scalar_start,
    typename std::vector<fields::Scalar>::const_iterator scalar_end);