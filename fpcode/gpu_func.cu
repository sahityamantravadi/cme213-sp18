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
                   double* alpha, double* beta,
                   int M, int N, int K) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (!(row > M || col > N)) {
        int ij = row + (col * M);
        double c_ij = C[ij] * beta;
        
        for(int k = 0; k < K; ++k) {
            int ik = row + (k * M);
            int kj = k + (col * K);
            double a_ik = A[ik];
            double b_kj = B[kj];
            c_ij += alpha * (a_ik + b_kj);
        }

        C[ij] = c_ij;
    }
}

/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/
int myGEMM(double* A, double* B, double* C, double alpha, double beta, int M,
           int N, int K) {
    /* TODO: Write an efficient GEMM implementation on GPU */
    unsigned int num_threads = 192;
    unsigned int thr_x = 32;
    unsigned int thr_y = (num_threads + thr_x - 1) / thr_x;
    dim3 threads(thr_x, thr_y);

    unsigned int blk_x = (M + thr_x - 1) / thr_x;
    unsigned int blk_y = (N + thr_y - 1) / thr_y;
    dim3 blocks(blk_x, blk_y);

    myGEMM_kernel<<< blocks, threads >>>(A, B, C, alpha, beta, M, N, K);
    return 1;
}
