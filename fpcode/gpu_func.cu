#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"
#include <cmath>

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
        A[ij] = 1 / (exp(-1.0 * A[ij]) + 1.0);
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

/* GPU kernel for summing rows of matrix A. Places row sums in vector v*/
__global__
void gpuRowSum_kernel(double *A, double *v, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (!(row > M)) {
        double rowSum = 0.0;
        int ind;
        for (int i = 0; i < N; i++) {
            ind = row + (M*i);
            rowSum += A[ind];
        }
        v[row] = rowSum;
    }
}

/* Routine for summing rows of matrix A. Places row sums in vector v */
void gpuRowSum(double *A, double *v, int M, int N) {
    unsigned int num_threads = 192;
    unsigned int thr_x = n_threads;
    dim3 threads(thr_x);

    unsigned int blk_x = (M + thr_x - 1) / thr_x;
    dim3 blocks(blk_x);

    gpuRowSum_kernel<<< blocks, threads >>>(A, v, M, N);
}

/* GPU kernel for broadcasting sum for matrix A with vector v */
__global__
void gpuMatVecSum_kernel(double *A, double *v, int M, int N) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;

    if (!(row > M || col > N)) {
        int ind = row + (M*col);
        double num = v[row];
        A[ind] += num;
    }
}

/* Routine for broadcasting sum for matrix A with vector v */
void gpuMatVecSum(double *A, double *v, int M, int N) {
    unsigned int num_threads = 192;
    unsigned int thr_x = 16;
    unsigned int thr_y = (num_threads + thr_x - 1) / thr_x;
    dim3 threads(thr_x, thr_y);

    unsigned int blk_x = (N + thr_x - 1) / thr_x;
    unsigned int blk_y = (num_neurons + thr_y - 1) / thr_y;
    dim3 blocks(blk_x, blk_y);

    gpuMatVecSum_kernel<<< blocks, threads >>>(A, v, M, N);
}

/* GPU kernel for elementwise Hadamard product */
__global__
void gpuHadamard_kernel(double *A, double *B, double *C, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (!(row > M || col > N)) {
        int ind = row + (M*col);
        C[ind] = A[ind] * B[ind];
    }
}

/* Routine for elementwise Hadamard product */
void gpuHadamard(double *A, double *B, double *C, int M, int N) {
    unsigned int num_threads = 192;
    unsigned int thr_x = 16;
    unsigned int thr_y = (num_threads + thr_x - 1) / thr_x;
    dim3 threads(thr_x, thr_y);

    unsigned int blk_x = (N + thr_x - 1) / thr_x;
    unsigned int blk_y = (num_neurons + thr_y - 1) / thr_y;
    dim3 blocks(blk_x, blk_y);

    gpuHadamard_kernel<<< blocks, threads >>>(A, B, C, M, N);
}

/* GPU kernel for elementwise matrix sum */
__global__
void gpuElementwiseSum_kernel(double *A, double *B, double *C,
                              double alpha, double beta,
                              int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (!(row > M || col > N)) {
        int ind = row + (M*col);
        C[ind] = (alpha * A[ind]) + (beta * B[ind]);
    }
}

/* Routine for elementwise matrix sum */
void gpuElementwiseSum(double *A, double *B, double *C, 
                       double alpha, double beta,
                       int M, int N) {
    unsigned int num_threads = 192;
    unsigned int thr_x = 16;
    unsigned int thr_y = (num_threads + thr_x - 1) / thr_x;
    dim3 threads(thr_x, thr_y);

    unsigned int blk_x = (N + thr_x - 1) / thr_x;
    unsigned int blk_y = (num_neurons + thr_y - 1) / thr_y;
    dim3 blocks(blk_x, blk_y);

    gpuElementwiseSum_kernel<<< blocks, threads >>>(A, B, C, alpha, beta, M, N);
}

/* GPU kernel for in-place matrix scalar prodcut */
__global__
void gpuMatrixScalarProduct_kernel(double *A, double alpha, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (!(row > M || col > N)) {
        int ind = row + (M*col);
        A[ind] = (alpha * A[ind]);
    }
}

/* Routine for in-place matrix scalar product */
void gpuMatrixScalarProduct(double *A, double alpha, int M, int N) {
    unsigned int num_threads = 192;
    unsigned int thr_x = 16;
    unsigned int thr_y = (num_threads + thr_x - 1) / thr_x;
    dim3 threads(thr_x, thr_y);

    unsigned int blk_x = (N + thr_x - 1) / thr_x;
    unsigned int blk_y = (num_neurons + thr_y - 1) / thr_y;
    dim3 blocks(blk_x, blk_y);

    gpuMatrixScalarProduct_kernel<<< blocks, threads >>>(A, alpha, M, N);
}

/* GPU kernel for derivative of sigmoid */
__global__
void gpudSigmoid_kernel(double *A, double *B, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (!(row > M || col > N)) {
        int ind = row + (M*col);
        A[ind] = A[ind] * B[ind] * (1.0 - B[ind]);
    }
}

void gpuFeedforward(device_cache &d, int N, NeuralNetwork &nn) {
    int num_neurons = d.num_neurons;
    int num_classes = d.num_classes;
    int num_pixels = d.num_pixels;

    double* one = 1.0;
    double* zero = 0.0;

    dim3 threads(thr_x, thr_y);
    dim3 blocks(blk_x, blk_y);
    
    // Computing activation from first layer
    myGEMM(d.W1, d.X, d.A1, one, zero, num_neurons, N, num_pixels, false, false, true);
    gpuMatVecSum(d.A1, d.b1, num_neurons, N);
    gpuSigmoid(d.A1, num_neurons, N);

    //Computing activation from second layer
    myGEMM(d.W2, d.A1, d.A2, one, zero, num_classes, N, num_neurons, false, false, true);
    gpuMatVecSum(d.A2, d.b2, num_classes, N);
    gpuSoftmax(d.A2, num_classes, N);
    d.yh = d.A2;
}

void gpuBackprop(device_cache &d, int N, double regularization, NeuralNetwork &nn, int num_processes) {
    int num_neurons = d.num_neurons;
    int num_classes = d.num_classes;
    int num_pixels = d.num_pixels;

    double one = 1.0;
    double zero = 0.0;
    double denom = (double) num_processes * N;
    double Ninv_pos = 1.0/(denom);
    double Ninv_neg = -1.0/(denom);
    double mod_reg = regularization/((double) num_processes);

    double* y_diff;
    cudaMalloc((void **) &y_diff,   sizeof(double) * num_classes * N);
    gpuElementwiseSum(y_diff, d.y, d.yh, Ninv_neg, Ninv_pos, num_classes, N);

    double* dW2_copy;
    cudaMalloc((void **) &dW2_copy, sizeof(double) * num_classes * num_neurons);
    cudaMemcpy(dW2_copy, nn.W[1].memptr(), sizeof(double) * num_classes * num_neurons, cudaMemcpyHostToDevice);

    myGEMM(y_diff, d.A1, dW2_copy, &one, &mod_reg, num_classes, num_neurons, N, false, true, false);

    cudaMemcpy(d.dW2, dW2_copy, sizeof(double) * num_classes * num_neurons, cudaMemcpyDeviceToHost);

    gpuRowSum(y_diff, d.db2, num_classes, N);

    myGEMM(d.W2, y_diff, d.dA1, &one, &zero, num_neurons, N, num_classes, true, false, true);




}

