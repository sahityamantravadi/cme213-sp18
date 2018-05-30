#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"

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
                   bool AT, bool BT, bool CZ) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (!(row > M || col > N)) {
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
        if (CZ)
            C[c_ind] = (alpha * dot_prod)
        else
            C[c_ind] = (alpha * dot_prod) + (beta * C[c_ind]);
    }
}

/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/
int myGEMM(double* A, double* B, double* C,
           double* alpha, double* beta,
           int M, int N, int K,
           bool AT = false, bool BT = false, bool CZ = false) {
    /* TODO: Write an efficient GEMM implementation on GPU */
    unsigned int num_threads = 192;
    unsigned int thr_x = 16;
    unsigned int thr_y = (num_threads + thr_x - 1) / thr_x;
    
    dim3 threads(thr_x, thr_y);

    unsigned int blk_x = (M + thr_x - 1) / thr_x;
    unsigned int blk_y = (N + thr_y - 1) / thr_y;
    dim3 blocks(blk_x, blk_y);

    myGEMM_kernel<<< blocks, threads >>>(A, B, C, *alpha, *beta, M, N, K, AT, BT, CZ);

    return 0;
}

/* GPU kernel for 10-class softmax */
__global__
void gpuSoftmax_kernel(double* A, unsigned int num_classes, unsigned int N) {
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (!(col > N)) {
        double denominator = 0.0;

        for(unsigned int c = 0; c < num_classes; c++){
            unsigned int ij = c + (col * num_classes);
            denominator += exp(A[ij]);
        }

        for(unsigned int c = 0; c < num_classes; c++){
            unsigned int ij = c + (col * num_classes);
            A[ij] = exp(A[ij]) / denominator;
        }
    }
}

/* Routine for 10-class softmax */
void gpuSoftmax(double* A, unsigned int num_classes, unsigned int N) {
    unsigned int num_threads = 192;
    unsigned int thr_x = 32;
    unsigned int thr_y = (num_threads + thr_x - 1) / thr_x;
    dim3 threads(thr_x, thr_y);

    unsigned int blk_x = (N + thr_x - 1) / thr_x;
    unsigned int blk_y = (num_classes + thr_y - 1) / thr_y;
    dim3 blocks(blk_x, blk_y);

    gpuSoftmax_kernel<<< blocks, threads >>>(A, num_classes, N);

}

/* GPU kernel for in-place element-wise sigmoid */
__global__
void gpuSigmoid_kernel(double* A, unsigned int num_neurons, unsigned int N) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(!(col > N || row > num_neurons)) {
        unsigned int ij = row + (col * num_neurons);
        A[ij] = 1 / (exp(-1.0 * A[ij]) + 1);
    }
}

/* Routine for in-place element-wise sigmoid */
void gpuSigmoid(double* A, unsigned int num_neurons, unsigned int N) {
    unsigned int num_threads = 192;
    unsigned int thr_x = 16;
    unsigned int thr_y = (num_threads + thr_x - 1) / thr_x;
    dim3 threads(thr_x, thr_y);

    unsigned int blk_x = (N + thr_x - 1) / thr_x;
    unsigned int blk_y = (num_neurons + thr_y - 1) / thr_y;
    dim3 blocks(blk_x, blk_y);

    gpuSigmoid_kernel<<< blocks, threads >>>(A, num_neurons, N);

}

void gpuFeedforward(device_cache &d, int N, NeuralNetwork &nn) {
    int num_neurons = d.num_neurons;
    int num_classes = d.num_classes;
    int num_pixels = d.num_pixels;
}
