#define GRID_SIZE 32
#define BLOCK_SIZE 16 

template<typename FieldT> 
__device__ __constant__ FieldT zero;

template <typename T, unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template<typename T>
__device__ T out;

template <typename T, typename FieldT, unsigned int blockSize>
__global__ void cuda_multi_exp_inner(T *vec, FieldT *scalar, unsigned int field_size) {
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;

    sdata[tid] = zero;
    while (i < field_size) { 
        sdata[tid] += scalar[i] * vec[i]; i += gridSize; }
    __syncthreads();
    
    if (blockSize >= 2048) { if (tid < 1024) { sdata[tid] += sdata[tid + 1024]; } __syncthreads(); }
    if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) out = sdata[0];
}

template<typename T, typename FieldT, multi_exp_method Method,
    typename std::enable_if<(Method == multi_exp_method_cuda), int>::type = 0>
T multi_exp_inner(
    typename std::vector<T>::const_iterator vec_start,
    typename std::vector<T>::const_iterator vec_end,
    typename std::vector<FieldT>::const_iterator scalar_start,
    typename std::vector<FieldT>::const_iterator scalar_end)
{
    T *d_vec;
    FieldT *d_scalar
    size_t vec_size = (vec_end - vec_start) * sizeof(T);
    size_t scalar_size = (scalar_end - scalar_start) * sizeof(FieldT);

    cudaMalloc((void **)&d_vec, vec_size);
    cudaMalloc((void **)&d_scalar, scalar_size;
	
    cudaMemcpy(d_vec, &vec_start, vec_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scalar, &scalar_start, scalar_size, cudaMemcpyHostToDevice);
    smemSize = vec_size / 2;	

    dim3 dimGrid (GRID_SIZE, GRID_SIZE);
    dim3 dimBlock (BLOCK_SIZE, BLOCK_SIZE);
    threads = dimGrid.x * dimGrid.y;
    switch (threads)
    {
        case 2048:
        cuda_multi_exp_inner<2048><<< dimGrid, dimBlock, smemSize >>>(d_vec, d_scalar); break;
        case 1024:
        cuda_multi_exp_inner<1024><<< dimGrid, dimBlock, smemSize >>>(d_vec, d_scalar); break;
        case 512:
        cuda_multi_exp_inner<512><<< dimGrid, dimBlock, smemSize >>>(d_vec, d_scalar); break;
        case 256:
        cuda_multi_exp_inner<256><<< dimGrid, dimBlock, smemSize >>>(d_vec, d_scalar); break;
    }

    T* res; T result;
    CUDA_CALL (cudaGetSymbolAddress((void**) &res, out<T>));
    cudaMemcpy(result, res, sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_vec);
    cudaFree(d_scalar);
    return result;
}