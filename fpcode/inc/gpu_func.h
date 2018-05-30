#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

struct event_pair {
    cudaEvent_t start;
    cudaEvent_t end;
};

inline void check_launch(const char* kernel_name) {
    cudaThreadSynchronize();
    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess) {
        std::cerr << "error in " << kernel_name << " kernel" << std::endl;
        std::cerr << "error was: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

inline void start_timer(event_pair* p) {
    cudaEventCreate(&p->start);
    cudaEventCreate(&p->end);
    cudaEventRecord(p->start, 0);
}


inline double stop_timer(event_pair* p) {
    cudaEventRecord(p->end, 0);
    cudaEventSynchronize(p->end);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, p->start, p->end);
    cudaEventDestroy(p->start);
    cudaEventDestroy(p->end);
    return elapsed_time;
}

int useless_gpu_add_one(int t);

int myGEMM(double* A, double* B, double* C,
           double* alpha, double* beta,
           int M, int N, int K,
           bool AT=false, bool BT=false, bool CZ=false);

struct device_cache {
    double *X, *y;
    double *a1, *da1, *a2;
    double *W1, *dW1, *W2, *dW2;
    double *b1, *db1, *b2, *db2;
    int batch_size, num_pixels, num_classes, num_neurons;
 
    //B is batch size
    //P is number of pixels
    //C is number of classes
    //N is number of neurons
    device_cache(int B, int P, int C, int N) : batch_size(B), num_pixels(P), num_classes(C), num_neurons(N) {
        cudaMalloc((void **) &X,    sizeof(double) * B * P);
        cudaMalloc((void **) &y,    sizeof(double) * B * C);
        cudaMalloc((void **) &a1,   sizeof(double) * B * N);
        cudaMalloc((void **) &da1,  sizeof(double) * B * N);
        cudaMalloc((void **) &a2,   sizeof(double) * B * C);
        cudaMalloc((void **) &W1,   sizeof(double) * N * P);
        cudaMalloc((void **) &dW1,  sizeof(double) * N * P);
        cudaMalloc((void **) &W2,   sizeof(double) * N * C);
        cudaMalloc((void **) &dW2,  sizeof(double) * N * C);
        cudaMalloc((void **) &b1,   sizeof(double) * N); 
        cudaMalloc((void **) &db1,  sizeof(double) * N); 
        cudaMalloc((void **) &b2,   sizeof(double) * C);
        cudaMalloc((void **) &db2,  sizeof(double) * C);
    }
    
    ~device_cache() {
        cudaFree(X);
        cudaFree(y);
        cudaFree(a1);
        cudaFree(da1);
        cudaFree(a2);
        cudaFree(W1);
        cudaFree(dW1);
        cudaFree(W2);
        cudaFree(dW2);
        cudaFree(b1);
        cudaFree(db1);
        cudaFree(b2);
        cudaFree(db2);
    }
}

void gpuSigmoid(double* A, unsigned int num_neurons, unsigned int N);
void gpuSoftmax(double* A, unsigned int num_classes, unsigned int N);

#endif
