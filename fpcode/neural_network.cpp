#include "neural_network.h"

#include <armadillo>
#include "utils/common.h"
#include "gpu_func.h"
#include "mpi.h"
#include "iomanip"

#define MPI_SAFE_CALL( call ) do {                               \
    int err = call;                                              \
    if (err != MPI_SUCCESS) {                                    \
        fprintf(stderr, "MPI error %d in file '%s' at line %i",  \
               err, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    } } while(0)

double norms(NeuralNetwork& nn) {
    double norm_sum = 0;

    for(int i = 0; i < nn.num_layers; ++i)  {
        norm_sum += arma::accu(arma::square(nn.W[i]));
    }

    return norm_sum;
}

void write_cpudata_tofile(NeuralNetwork& nn, int iter) {
    std::stringstream s;
    s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
    nn.W[0].save(s.str(), arma::raw_ascii);
    std::stringstream t;
    t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
    nn.W[1].save(t.str(), arma::raw_ascii);
    std::stringstream u;
    u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
    nn.b[0].save(u.str(), arma::raw_ascii);
    std::stringstream v;
    v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
    nn.b[1].save(v.str(), arma::raw_ascii);
}

void write_diff_gpu_cpu(NeuralNetwork& nn, int iter,
                        std::ofstream& error_file) {
    arma::mat A, B, C, D;

    std::stringstream s;
    s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
    A.load(s.str(), arma::raw_ascii);
    double max_errW0 = arma::norm(nn.W[0]-A, "inf")/arma::norm(A, "inf");
    double L2_errW0  = arma::norm(nn.W[0]-A,2)/arma::norm(A,2);

    std::stringstream t;
    t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
    B.load(t.str(), arma::raw_ascii);
    double max_errW1 = arma::norm(nn.W[1]-B, "inf")/arma::norm(B, "inf");
    double L2_errW1  = arma::norm(nn.W[1]-B,2)/arma::norm(B,2);

    std::stringstream u;
    u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
    C.load(u.str(), arma::raw_ascii);
    double max_errb0 = arma::norm(nn.b[0]-C, "inf")/arma::norm(C, "inf");
    double L2_errb0  = arma::norm(nn.b[0]-C,2)/arma::norm(C,2);

    std::stringstream v;
    v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
    D.load(v.str(), arma::raw_ascii);
    double max_errb1 = arma::norm(nn.b[1]-D, "inf")/arma::norm(D, "inf");
    double L2_errb1  = arma::norm(nn.b[1]-D,2)/arma::norm(D,2);

    int ow = 15;

    if(iter == 0) {
        error_file << std::left<< std::setw(ow) << "Iteration" << std::left<< std::setw(
                       ow) << "Max Err W0" << std::left << std::setw(ow) << "Max Err W1"
                   << std::left<< std::setw(ow) << "Max Err b0" << std::left<< std::setw(
                       ow) << "Max Err b1" << std::left << std::setw(ow) << "L2 Err W0" << std::left
                   << std::setw(ow) << "L2 Err W1" << std::left<< std::setw(
                       ow) << "L2 Err b0" << std::left<< std::setw(ow) << "L2 Err b1" << "\n";
    }

    error_file << std::left << std::setw(ow) << iter << std::left << std::setw(
                   ow) << max_errW0 << std::left << std::setw(ow) << max_errW1 <<
               std::left << std::setw(ow) << max_errb0 << std::left << std::setw(
                   ow) << max_errb1 << std::left<< std::setw(ow) << L2_errW0 << std::left <<
               std::setw(ow) << L2_errW1 << std::left << std::setw(ow) << L2_errb0 <<
               std::left<< std::setw(ow) << L2_errb1 << "\n";

}

/* CPU IMPLEMENTATIONS */
void feedforward(NeuralNetwork& nn, const arma::mat& X, struct cache& cache) {
    cache.z.resize(2);
    cache.a.resize(2);

    // std::cout << W[0].n_rows << "\n";tw
    assert(X.n_rows == nn.W[0].n_cols);
    cache.X = X;
    int N = X.n_cols;

    arma::mat z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N);
    cache.z[0] = z1;

    arma::mat a1;
    sigmoid(z1, a1);
    cache.a[0] = a1;

    assert(a1.n_rows == nn.W[1].n_cols);
    arma::mat z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
    cache.z[1] = z2;

    arma::mat a2;
    softmax(z2, a2);
    cache.a[1] = cache.yc = a2;
}

/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : C x N one-hot column vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop(NeuralNetwork& nn, const arma::mat& y, double reg,
              const struct cache& bpcache, struct grads& bpgrads) {
    bpgrads.dW.resize(2);
    bpgrads.db.resize(2);
    int N = y.n_cols;

    // std::cout << "backprop " << bpcache.yc << "\n";
    arma::mat diff = (1.0 / N) * (bpcache.yc - y);
    bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
    bpgrads.db[1] = arma::sum(diff, 1);
    arma::mat da1 = nn.W[1].t() * diff;

    arma::mat dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

    bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
    bpgrads.db[0] = arma::sum(dz1, 1);
}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
double loss(NeuralNetwork& nn, const arma::mat& yc, const arma::mat& y,
            double reg) {
    int N = yc.n_cols;
    double ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));

    double data_loss = ce_sum / N;
    double reg_loss = 0.5 * reg * norms(nn);
    double loss = data_loss + reg_loss;
    // std::cout << "Loss: " << loss << "\n";
    return loss;
}

/*
 * Returns a vector of labels for each row vector in the input
 */
void predict(NeuralNetwork& nn, const arma::mat& X, arma::rowvec& label) {
    struct cache fcache;
    feedforward(nn, X, fcache);
    label.set_size(X.n_cols);

    for(int i = 0; i < X.n_cols; ++i) {
        arma::uword row;
        fcache.yc.col(i).max(row);
        label(i) = row;
    }
}

/*
 * Computes the numerical gradient
 */
void numgrad(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
             double reg, struct grads& numgrads) {
    double h = 0.00001;
    struct cache numcache;
    numgrads.dW.resize(nn.num_layers);
    numgrads.db.resize(nn.num_layers);

    for(int i = 0; i < nn.num_layers; ++i) {
        numgrads.dW[i].resize(nn.W[i].n_rows, nn.W[i].n_cols);

        for(int j = 0; j < nn.W[i].n_rows; ++j) {
            for(int k = 0; k < nn.W[i].n_cols; ++k) {
                double oldval = nn.W[i](j,k);
                nn.W[i](j, k) = oldval + h;
                feedforward(nn, X, numcache);
                double fxph = loss(nn, numcache.yc, y, reg);
                nn.W[i](j, k) = oldval - h;
                feedforward(nn, X, numcache);
                double fxnh = loss(nn, numcache.yc, y, reg);
                numgrads.dW[i](j, k) = (fxph - fxnh) / (2*h);
                nn.W[i](j, k) = oldval;
            }
        }
    }

    for(int i = 0; i < nn.num_layers; ++i) {
        numgrads.db[i].resize(nn.b[i].n_rows, nn.b[i].n_cols);

        for(int j = 0; j < nn.b[i].size(); ++j) {
            double oldval = nn.b[i](j);
            nn.b[i](j) = oldval + h;
            feedforward(nn, X, numcache);
            double fxph = loss(nn, numcache.yc, y, reg);
            nn.b[i](j) = oldval - h;
            feedforward(nn, X, numcache);
            double fxnh = loss(nn, numcache.yc, y, reg);
            numgrads.db[i](j) = (fxph - fxnh) / (2*h);
            nn.b[i](j) = oldval;
        }
    }
}

/*
 * Train the neural network &nn
 */
void train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
           double learning_rate, double reg,
           const int epochs, const int batch_size, bool grad_check, int print_every,
           int debug) {
    int N = X.n_cols;
    int iter = 0;
    int print_flag = 0;

    for(int epoch = 0 ; epoch < epochs; ++epoch) {
        int num_batches = (N + batch_size - 1)/batch_size;

        for(int batch = 0; batch < num_batches; ++batch) {
            int last_col = std::min((batch + 1)*batch_size-1, N-1);
            arma::mat X_batch = X.cols(batch * batch_size, last_col);
            arma::mat y_batch = y.cols(batch * batch_size, last_col);

            struct cache bpcache;
            feedforward(nn, X_batch, bpcache);

            struct grads bpgrads;
            backprop(nn, y_batch, reg, bpcache, bpgrads);

            if(print_every > 0 && iter % print_every == 0) {
                if(grad_check) {
                    struct grads numgrads;
                    numgrad(nn, X_batch, y_batch, reg, numgrads);
                    assert(gradcheck(numgrads, bpgrads));
                }

                std::cout << "Loss at iteration " << iter << " of epoch " << epoch << "/" <<
                          epochs << " = " << loss(nn, bpcache.yc, y_batch, reg) << "\n";
            }

            // Gradient descent step
            for(int i = 0; i < nn.W.size(); ++i) {
                nn.W[i] -= learning_rate * bpgrads.dW[i];
            }

            for(int i = 0; i < nn.b.size(); ++i) {
                nn.b[i] -= learning_rate * bpgrads.db[i];
            }

            /* Debug routine runs only when debug flag is set. If print_every is zero, it saves
               for the first batch of each epoch to avoid saving too many large files.
               Note that for the first time, you have to run debug and serial modes together.
               This will run the following function and write out files to CPUmats folder.
               In the later runs (with same parameters), you can use just the debug flag to
               output diff b/w CPU and GPU without running CPU version */
            if(print_every <= 0) {
                print_flag = batch == 0;
            } else {
                print_flag = iter % print_every == 0;
            }

            if(debug && print_flag) {
                write_cpudata_tofile(nn, iter);
            }

            iter++;
        }
    }
}



void gpuFeedforward(device_cache &d, int N) {
    int num_neurons = d.num_neurons;
    int num_classes = d.num_classes;
    int num_pixels = d.num_pixels;
    double one = 1.0;
    double zero = 0.0;
    
    // Computing activation from first layer
    myGEMM(d.W1, d.X, d.A1, &one, &zero, num_neurons, N, num_pixels, false, false, true);
    gpuMatVecSum(d.A1, d.b1, num_neurons, N);
    gpuSigmoid(d.A1, num_neurons, N);
    
    // Computing activation from second layer
    myGEMM(d.W2, d.A1, d.A2, &one, &zero, num_classes, N, num_neurons, false, false, true);
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

    double *y_diff;
    cudaMalloc((void **) &y_diff, sizeof(double) * num_classes * N);
    gpuElementwiseSum(y_diff, d.y, d.yh, Ninv_neg, Ninv_pos, num_classes, N);

    double *dW2_copy;
    cudaMalloc((void **) &dW2_copy, sizeof(double) * num_classes * num_neurons);
    cudaMemcpy(dW2_copy, nn.W[1].memptr(), sizeof(double) * num_classes * num_neurons, cudaMemcpyHostToDevice);

    myGEMM(y_diff, d.A1, dW2_copy, &one, &mod_reg, num_classes, num_neurons, N, false, true, false);

    cudaMemcpy(d.dW2, dW2_copy, sizeof(double) * num_classes * num_neurons, cudaMemcpyDeviceToHost);

    gpuRowSum(y_diff, d.db2, num_classes, N);

    myGEMM(d.W2, y_diff, d.dA1, &one, &zero, num_neurons, N, num_classes, true, false, true);

    double *dZ1;
    cudaMalloc((void **) &dZ1, sizeof(double) * num_neurons * N);
    gpudSigmoid(d.dA1, d.A1, dZ1, num_neurons, N);

    myGEMM(dZ1, d.X, d.W1, &one, &mod_reg, num_neurons, num_pixels, N, false, true, false);
    d.dW1 = d.W1;

    gpuRowSum(d.db1, dZ1, num_neurons, N);

    cudaFree(y_diff);
    cudaFree(dW2_copy);
    cudaFree(dZ1);
}



/*
 * TODO
 * Train the neural network &nn of rank 0 in parallel. Your MPI implementation
 * should mainly be in this function.
 */
void parallel_train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
                    double learning_rate, double reg,
                    const int epochs, const int batch_size, bool grad_check, int print_every,
                    int debug) {

    int rank, num_procs;
    MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
    MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    int N = (rank == 0)?X.n_cols:0;
    MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));

    std::ofstream error_file;
    error_file.open("Outputs/CpuGpuDiff.txt");
    int print_flag = 0;

    /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
       for storing elements in a column major way. Or you can allocate your own array
       memory space and store the elements in a row major way. Remember to update the
       Armadillo matrices in NeuralNetwork &nn of rank 0 before returning from the function. */
    int num_pixels = nn.H[0];
    int num_neurons = nn.H[1];
    int num_classes = nn.H[2];
    /* iter is a variable used to manage debugging. It increments in the inner loop
       and therefore goes from 0 to epochs*num_batches */
    int iter = 0;

    double *host_dW1 = (double *) malloc(sizeof(double) * num_neurons * num_pixels);
    double *host_dW2 = (double *) malloc(sizeof(double) * num_classes * num_neurons);
    double *host_db1 = (double *) malloc(sizeof(double) * num_neurons);
    double *host_db2 = (double *) malloc(sizeof(double) * num_classes);

    double *host_dW1_l = (double *) malloc(sizeof(double) * num_neurons * num_pixels);
    double *host_dW2_l = (double *) malloc(sizeof(double) * num_classes * num_neurons);
    double *host_db1_l = (double *) malloc(sizeof(double) * num_neurons);
    double *host_db2_l = (double *) malloc(sizeof(double) * num_classes);


    device_cache d(batch_size, num_pixels, num_classes, num_neurons);

    double *X_batch = (double *) malloc(sizeof(double) * num_pixels * batch_size);
    double *y_batch = (double *) malloc(sizeof(double) * num_classes * batch_size);

    // adjust learning rate and regularization for number of processes
    learning_rate = learning_rate/num_procs;
    reg = reg/num_procs;

    for(int epoch = 0; epoch < epochs; ++epoch) {
        int num_batches = (N + batch_size - 1)/batch_size;

        for(int batch = 0; batch < num_batches; ++batch) {
            /*
             * Possible Implementation:
             * 1. subdivide input batch of images and `MPI_scatter()' to each MPI node
             * 2. compute each sub-batch of images' contribution to network coefficient updates
             * 3. reduce the coefficient updates and broadcast to all nodes with `MPI_Allreduce()'
             * 4. update local network coefficient at each node
             */
            int start_point = (batch * batch_size);
            int end_point = std::min((batch + 1) * batch_size - 1, N - 1);
            int in_proc = std::min(batch_size, N - start_point)/num_procs;

            MPI_SAFE_CALL(
                MPI_Scatter( 
                    X.colptr(start_point),
                    num_pixels * in_proc,
                    MPI_DOUBLE,
                    X_batch,
                    num_pixels * in_proc,
                    MPI_DOUBLE,
                    0, 
                    MPI_COMM_WORLD
            ));
            MPI_SAFE_CALL(
                MPI_Scatter(
                    y.colptr(start_point),
                    num_classes * in_proc,
                    MPI_DOUBLE,
                    y_batch_proc,
                    num_classes * in_proc,
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD
            ));

            //data transfer
            cudaMemcpy(d.X, X_batch, sizeof(double) * num_pixels * in_proc, cudaMemcpyHostToDevice);
            cudaMemcpy(d.y, y_batch, sizeof(double) * num_classes * in_proc, cudaMemcpyHostToDevice);
            cudaMemcpy(d.W1, nn.W[0].memptr(), sizeof(double) * num_pixels * num_neurons, cudaMemcpyHostToDevice);
            cudaMemcpy(d.b1, nn.b[0].memptr(), sizeof(double) * num_neurons, cudaMemcpyHostToDevice);
            cudaMemcpy(d.W2, nn.W[1].memptr(), sizeof(double) * num_neurons * num_classes, cudaMemcpyHostToDevice);
            cudaMemcpy(d.b2, nn.b[1].memptr(), sizeof(double) * num_classes, cudaMemcpyHostToDevice);

            gpuFeedforward(d, in_proc, nn);

            gpuBackprop(d, in_proc, reg, nn, num_procs);

            cudaMemcpy(host_dW1, d.dW1, sizeof(double) * num_pixels * num_neurons, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_db1, d.db1, sizeof(double) * num_neurons, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_dW2, d.dW2, sizeof(double) * num_neurons * num_classes, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_db2, d.db2, sizeof(double) * num_classes, cudaMemcpyDeviceToHost);

            MPI_SAFE_CALL(
                MPI_Allreduce(
                    host_dW1,
                    host_dW1_l,
                    num_pixels * num_neurons,
                    MPI_DOUBLE,
                    MPI_SUM,
                    MPI_COMM_WORLD
            ));
            MPI_SAFE_CALL(
                MPI_Allreduce(
                    host_db1,
                    host_db1_l,
                    num_neurons,
                    MPI_DOUBLE,
                    MPI_SUM,
                    MPI_COMM_WORLD
            ));
            MPI_SAFE_CALL(
                MPI_Allreduce(
                    host_dW2,
                    host_dW2_l,
                    num_neurons * num_classes,
                    MPI_DOUBLE,
                    MPI_SUM,
                    MPI_COMM_WORLD
            ));
            MPI_SAFE_CALL(
                MPI_Allreduce(
                    host_db2,
                    host_db2_l,
                    num_classes,
                    MPI_DOUBLE,
                    MPI_SUM,
                    MPI_COMM_WORLD
            ));

            //gradient descent
            nn.W[0] -= learning_rate * arma::mat(h_dW1_l, nn.W[0].n_rows, nn.W[0].n_cols);
            nn.b[0] -= learning_rate * arma::mat(h_db1_l, nn.b[0].n_rows, nn.b[0].n_cols);
            nn.W[1] -= learning_rate * arma::mat(h_dW2_l, nn.W[1].n_rows, nn.W[1].n_cols);
            nn.b[1] -= learning_rate * arms::mat(h_db2_l, nn.b[1].n_rows, nn.b[1].n_cols);

            if(print_every <= 0) {
                print_flag = batch == 0;
            } else {
                print_flag = iter % print_every == 0;
            }

            /* Following debug routine assumes that you have already updated the arma
               matrices in the NeuralNetwork nn.  */
            if(debug && rank == 0 && print_flag) {
                write_diff_gpu_cpu(nn, iter, error_file);
            }

            iter++;
        }
    }

    error_file.close();
}
