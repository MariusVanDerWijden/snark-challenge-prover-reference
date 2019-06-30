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

#define FULL_MASK 0xffffffff

namespace cuda {

template <typename T>
__inline__ __device__
T warpReduceSum(T val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) 
        val += T::shuffle_down(FULL_MASK, val, offset); 
    return val;
}

template <typename T>
__inline__ __device__
T blockReduceSum(T val) {
    static __shared__ T sMem[32];
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    val = warpReduceSum(val); 
    if (lane==0) sMem[warpId]=val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / warpSize) ? sMem[lane] : T::zero();
    if (warpId==0) val = warpReduceSum(val);
    return val;
}

template <typename T, typename FieldT>
__global__ void deviceReduceKernel(T *vec, FieldT *scalar, T *result, int field_size) {
    T sum = T::zero();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < field_size; i += blockDim.x * gridDim.x) {       
        sum += vec[i] * scalar[i];
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x==0)
        result[blockIdx.x] = sum;
}

template <typename T>
__global__ void deviceReduceKernelSecond(T *resIn, T *resOut, int field_size) {
    T sum = T::zero();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < field_size; i += blockDim.x * gridDim.x) {       
        sum += resIn[i];
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x==0)
        resOut[0] = sum;
}

template<typename T, typename FieldT> 
void toGPUField (
    typename std::vector<T>::const_iterator vec_start,
    typename std::vector<T>::const_iterator vec_end,
    typename std::vector<FieldT>::const_iterator scalar_start,
    typename std::vector<FieldT>::const_iterator scalar_end, 
    fields::mnt4753_G2 *d_vec,
    fields::Scalar *d_scalar) 
{
    size_t vec_size = (vec_end - vec_start);
    size_t scalar_size = (scalar_end - scalar_start);

    cudaMalloc((void **)&d_vec, vec_size * sizeof(fields::mnt4753_G2));
    cudaMalloc((void **)&d_scalar, scalar_size * sizeof(fields::Scalar));

    fields::mnt4753_G2 tmp_vec[vec_size];
    fields::Scalar tmp_scalar[scalar_size];
#ifdef MULTICORE
    #pragma omp parallel for
#endif
    for (uint i = 0; i < vec_size; i++) {
#ifndef BINARY_OUTPUT
#define BINARY_OUTPUT
#endif
        fields::mnt4753_G2 f;
        f.x << vec_start[i]->X().as_bigint();
        f.y << vec_start[i]->Y().as_bigint();
        tmp_vec[i] = f;

        fields::Scalar s;
        s << scalar_start[i]->as_bigint();
        tmp_scalar[i] = s;
    }
    cudaMemcpy(d_vec, &tmp_vec, vec_size * sizeof(fields::mnt4753_G2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scalar, &tmp_scalar, scalar_size * sizeof(fields::Scalar), cudaMemcpyHostToDevice);
}

fields::mnt4753_G2 startKernel(fields::mnt4753_G2 *d_vec, fields::Scalar *d_scalar, int length)
{
    fields::mnt4753_G2 *result;
    int threads = 512;
    int blocks = min((length + threads - 1) / threads, 1024);
    cudaMalloc(&result, sizeof(fields::mnt4753_G2) * 1024);

    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    //two runs of the kernel, better efficiency
    deviceReduceKernel<<<blocks, threads>>>(d_vec, d_scalar, result, length);
    deviceReduceKernelSecond<<<1, 1024>>>(result, result, blocks);

    fields::mnt4753_G2 res;
    cudaMemcpy(&res, result, sizeof(fields::mnt4753_G2), cudaMemcpyDeviceToHost);

    cudaFree(result);
    cudaFree(d_vec);
    cudaFree(d_scalar);
    return *result;
}

template<typename T, typename FieldT>
T cuda_multi_exp_inner(
    typename std::vector<T>::const_iterator vec_start,
    typename std::vector<T>::const_iterator vec_end,
    typename std::vector<FieldT>::const_iterator scalar_start,
    typename std::vector<FieldT>::const_iterator scalar_end)
{
    printf("Enter custom gpu code");

    fields::mnt4753_G2 *d_vec;
    fields::Scalar  *d_scalar;

    toGPUField(vec_start, vec_end, scalar_start, scalar_end, d_vec, d_scalar);
	
    fields::mnt4753_G2 res = startKernel(d_vec, d_scalar, (vec_end - vec_start));

    return res;
}
}