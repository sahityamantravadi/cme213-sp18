#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"
#include <cmath>

#define BLOCK_SIZE 32

__global__
void device_add_one(int* d_result, int t) {
    *d_result = t + 1;
}

/*
Just a dummy function that can be used to warm up GPU
*/
int useless_gpu_add_one(int t) {
    int result;
    int* d_result;

    checkCudaErrors(cudaMalloc((void**)&d_result, 1 * sizeof(int)));

    event_pair timer;
    start_timer(&timer);
    device_add_one<<<1,1>>>(d_result, t);
    check_launch("device_add_one");
    double time = stop_timer(&timer);

    std::cout << "device_add_one took: " << time << " seconds" << std::endl;

    checkCudaErrors(cudaMemcpy(&result, d_result, 1 * sizeof(int),
                               cudaMemcpyDeviceToHost));
    return result;
}

/* GPU kernel for in-place GEMM operation */
__global__
void myGEMM_kernel(double* A, double* B, double* C,
                   double alpha, double beta,
                   int M, int N, int K,
                   bool AT, bool BT) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        int c_ind = row + (col * M);
        double dot_prod = 0.0;
        int a_ind;
        int b_ind;
        for(int i = 0; i < K; i++) {
            if (AT)
                a_ind = (row*K) + i;
            else
                a_ind = row + (i*M);
            if (BT)
                b_ind = col + (i*N);
            else
                b_ind = i + (col * K);
            dot_prod += A[a_ind] * B[b_ind];
        }
        C[c_ind] = (alpha * dot_prod) + (beta * C[c_ind]);
    }
}

/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/
/*
   int myGEMM(double* A, double* B, double* C,
           double* alpha, double* beta,
           int M, int N, int K,
           bool AT, bool BT) {
    unsigned int num_threads = 192;
    unsigned int thr_x = 32;
    unsigned int thr_y = (num_threads + thr_x - 1) / thr_x;
    
    dim3 threads(thr_x, thr_y);

    unsigned int blk_x = (M + thr_x - 1) / thr_x;
    unsigned int blk_y = (N + thr_y - 1) / thr_y;
    dim3 blocks(blk_x, blk_y);

    myGEMM_kernel<<< blocks, threads >>>(A, B, C, *alpha, *beta, M, N, K, AT, BT);
    check_launch("myGEMM_kernel");
    return 0;
}
*/

__global__
void shared_myGEMM_kernel(double* __restrict__ A, double* __restrict__ B, double* __restrict__ C,
                          double alpha, double beta,
                          int M, int N, int K,
                          bool AT, bool BT) {
    int y_block = blockIdx.y;
    int x_block = blockIdx.x;
    int col = threadIdx.y;
    int row = threadIdx.x;
    __shared__ double A_shared[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ double B_shared[BLOCK_SIZE][BLOCK_SIZE + 1];

    double C_aggr = 0.0;
    unsigned int i_lim = (K + BLOCK_SIZE - 1)/BLOCK_SIZE;
    for (unsigned int i = 0; i < i_lim; ++i) {
        //fill in shared memory
        if (AT && ((BLOCK_SIZE * i) < (K - col))) {
            double *A_part = A + (K + BLOCK_SIZE * x_block + BLOCK_SIZE * i);
            unsigned int a_ind = (row * K) + col;
            A_shared[row][col] = A_part[a_ind];
        }
        else if ((BLOCK_SIZE * i) < (K - col)) {
            double *A_part = A + (M * BLOCK_SIZE * i + BLOCK_SIZE * x_block);
            unsigned int a_ind = (col * M) + row;
            A_shared[row][col] = A_part[a_ind];
        }
        else {
            A_shared[row][col] = 0.0;
        }

        if (BT && ((BLOCK_SIZE * i) < (K - row))) {
            double *B_part = B + (N * BLOCK_SIZE * i + BLOCK_SIZE * y_block);
            unsigned int b_ind = (row * N) + col;
            B_shared[row][col] = B_part[b_ind];
        }
        else if ((BLOCK_SIZE * i) < (K-row)) {
            double *B_part = B + (K * BLOCK_SIZE * y_block + BLOCK_SIZE * i);
            unsigned int b_ind = (col * K) +  row;
            B_shared[row][col] = B_part[b_ind];
        }
        else{
            B_shared[row][col] = 0.0;
        }

        __syncthreads();

        //matrix multiplication
        for (unsigned int j = 0; j < BLOCK_SIZE; ++j) {
            C_aggr += A_shared[row][j] * B_shared[j][col];
        }

        __syncthreads();
    }

    C_aggr *= alpha;

    //copy memory back
    int cur_col = BLOCK_SIZE * y_block + col;
    int cur_row = BLOCK_SIZE * x_block + row;
    if ((cur_col < N) && (cur_row < M)) {
        double *C_shared = C + (M * BLOCK_SIZE * y_block + BLOCK_SIZE * x_block);
        unsigned int c_ind = row + col * M;
        C_aggr += beta * C_shared[c_ind];
        C_shared[c_ind] = C_aggr;
    }
}

/* Routine to perform shared GEMM operation */
int myGEMM(double *A, double *B, double *C,
           double *alpha, double *beta,
           int M, int N, int K,
           bool AT, bool BT) {
    dim3 block_dims (BLOCK_SIZE, BLOCK_SIZE);
    int blk_x = (M + block_dims.x - 1)/block_dims.x;
    int blk_y = (N + block_dims.y - 1)/block_dims.y;

    dim3 grid_dims (blk_x, blk_y);

    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    shared_myGEMM_kernel <<<grid_dims, block_dims>>> (A, B, C, *alpha, *beta, M, N, K, AT, BT);
    check_launch("shared_myGEMM_kernel");
    return 0;
}


/* GPU kernel for 10-class softmax */
__global__
void gpuSoftmax_kernel(double* A, unsigned int num_classes, unsigned int N) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (col < N) {
        double denominator = 0.0;

        for(int c = 0; c < num_classes; c++){
            denominator += (double) std::exp(A[col*num_classes + c]);
        }

        for(int c = 0; c < num_classes; c++){
            int ij = c + (col * num_classes);
            A[ij] = (double) std::exp(A[ij])/ (double) denominator;
        }
    }
}

/* Routine for 10-class softmax */
void gpuSoftmax(double* A, unsigned int num_classes, unsigned int N) {
    unsigned int num_threads = 192;
    unsigned int thr_x = num_threads;
    dim3 threads(thr_x);

    unsigned int blk_x = (N + thr_x - 1) / thr_x;
    dim3 blocks(blk_x);

    gpuSoftmax_kernel<<< blocks, threads >>>(A, num_classes, N);
    check_launch("gpuSoftmax_kernel");
}

/* GPU kernel for in-place element-wise sigmoid */
__global__
void gpuSigmoid_kernel(double* A, unsigned int num_neurons, unsigned int N) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(col < N && row < num_neurons) {
        int ij = row + (col * num_neurons);
        A[ij] = (double) 1.0 / (double)(1.0 + exp(-1.0 * A[ij]));
    }
}

/* Routine for in-place element-wise sigmoid */
void gpuSigmoid(double* A, unsigned int num_neurons, unsigned int N) {
    unsigned int num_threads = 192;
    unsigned int thr_x = 32;
    unsigned int thr_y = (num_threads + thr_x - 1) / thr_x;
    dim3 threads(thr_x, thr_y);

    unsigned int blk_x = (num_neurons + thr_x - 1) / thr_x;
    unsigned int blk_y = (N + thr_y - 1) / thr_y;
    dim3 blocks(blk_x, blk_y);

    gpuSigmoid_kernel<<< blocks, threads >>>(A, num_neurons, N);
    check_launch("gpuSigmoid_kernel");
}

/* GPU kernel for summing rows of matrix A. Places row sums in vector v*/
__global__
void gpuRowSum_kernel(double *A, double *v, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        double rowSum = 0.0;
        for (int i = 0; i < N; i++) 
            rowSum += A[(M*i) + row];
        v[row] = rowSum;
    }
}

/* Routine for summing rows of matrix A. Places row sums in vector v */
void gpuRowSum(double *A, double *v, int M, int N) {
    unsigned int num_threads = 192;
    unsigned int thr_x = num_threads;
    dim3 threads(thr_x);

    unsigned int blk_x = (M + thr_x - 1) / thr_x;
    dim3 blocks(blk_x);

    gpuRowSum_kernel<<< blocks, threads >>>(A, v, M, N);
    check_launch("gpuRowSum_kernel");
}

/* GPU kernel for broadcasting sum for matrix A with vector v */
__global__
void gpuMatVecSum_kernel(double *A, double *v, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; 
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M &&  col < N) {
        int ind = row + (M*col);
        double num = v[row];
        A[ind] += num;
    }
}

/* Routine for broadcasting sum for matrix A with vector v */
void gpuMatVecSum(double *A, double *v, int M, int N) {
    unsigned int num_threads = 192;
    unsigned int thr_x = 32;
    unsigned int thr_y = (num_threads + thr_x - 1) / thr_x;
    dim3 threads(thr_x, thr_y);

    unsigned int blk_x = (M + thr_x - 1) / thr_x;
    unsigned int blk_y = (N + thr_y - 1) / thr_y;
    dim3 blocks(blk_x, blk_y);
    gpuMatVecSum_kernel<<< blocks, threads >>>(A, v, M, N);
    check_launch("gpuMatVecSum_kernel");
}

/* GPU kernel for elementwise matrix sum */
__global__
void gpuElementwiseSum_kernel(double *A, double *B, double *C,
                              double alpha, double beta,
                              int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < M && col < N) {
        int ind = row + (M*col);
        C[ind] = (alpha * A[ind]) + (beta * B[ind]);
    }
}

/* Routine for elementwise matrix sum */
void gpuElementwiseSum(double *A, double *B, double *C, 
                       double alpha, double beta,
                       int M, int N) {
    unsigned int num_threads = 192;
    unsigned int thr_x = 32;
    unsigned int thr_y = (num_threads + thr_x - 1) / thr_x;
    dim3 threads(thr_x, thr_y);

    unsigned int blk_x = (N + thr_x - 1) / thr_x;
    unsigned int blk_y = (M + thr_y - 1) / thr_y;
    dim3 blocks(blk_x, blk_y);

    gpuElementwiseSum_kernel<<< blocks, threads >>>(A, B, C, alpha, beta, M, N);
    check_launch("gpuElementwiseSum_kernel");
}

/* GPU kernel for derivative of sigmoid */
__global__
void gpudSigmoid_kernel(double *A, double *B, double *C, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        int ind = row + (M*col);
        C[ind] = (double) A[ind] * B[ind] * (1.0 - B[ind]);
    }
}

/** Routine for derivative of sigmoid */
void gpudSigmoid(double *A, double *B, double *C, int M, int N) {
    unsigned int num_threads = 192;
    unsigned int thr_x = 32;
    unsigned int thr_y = (num_threads + thr_x - 1) / thr_x;
    dim3 threads(thr_x, thr_y);

    unsigned int blk_x = (M + thr_x - 1) / thr_x;
    unsigned int blk_y = (N + thr_y - 1) / thr_y;
    dim3 blocks(blk_x, blk_y);

    gpudSigmoid_kernel<<< blocks, threads >>>(A, B, C, M, N);
    check_launch("gpudSigmoid");
}
