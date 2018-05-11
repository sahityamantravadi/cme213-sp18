/* Each kernel handles the update of one pagerank score. In other
 * words, each kernel handles one row of the update:
 *
 *      pi(t+1) = (1/2) A pi(t) + (1 / (2N))
 *      
 * You may assume that num_nodes <= blockDim.x * 65535
 *
 */
__global__
void device_graph_propagate(const uint* graph_indices
                            , const uint* graph_edges
                            , const float* graph_nodes_in
                            , float* graph_nodes_out
                            , const float* inv_edges_per_node
                            , int num_nodes) {
    // TODO: fill in the kernel code here
    const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint stride = blockDim.x*gridDim.x;

    for(uint i = tid; i < num_nodes; i += stride) {
        float sum = 0.f;

        for(uint j = graph_indices[i]; j < graph_indices[i+1]; ++j)
            sum += graph_nodes_in[graph_edges[j]]*inv_edges_per_node[graph_edges[j]];

        graph_nodes_out[i] = 0.5f/(float)num_nodes + 0.5f*sum;
    }
}

/* This function executes a specified number of iterations of the
 * pagerank algorithm. The variables are:
 *
 * h_graph_indices, h_graph_edges:
 *     These arrays describe the indices of the neighbors of node i.
 *     Specifically, node i is adjacent to all nodes in the range
 *     h_graph_edges[h_graph_indices[i] ... h_graph_indices[i+1]].
 *
 * h_node_values_input:
 *     An initial guess of pi(0).
 *
 * h_gpu_node_values_output:
 *     Output array for the pagerank vector.
 *
 * h_inv_edges_per_node:
 *     The i'th element in this array is the reciprocal of the
 *     out degree of the i'th node.
 *
 * nr_iterations:
 *     The number of iterations to run the pagerank algorithm for.
 *
 * num_nodes:
 *     The number of nodes in the whole graph (ie N).
 *
 * avg_edges:
 *     The average number of edges in the graph. You are guaranteed
 *     that the whole graph has num_nodes * avg_edges edges.
 *
 */
double device_graph_iterate(const uint* h_graph_indices
                            , const uint* h_graph_edges
                            , const float* h_node_values_input
                            , float* h_gpu_node_values_output
                            , const float* h_inv_edges_per_node
                            , int nr_iterations
                            , int num_nodes
                            , int avg_edges) {
    // TODO: allocate GPU memory
    //allocating GPU memory for each variable
    uint* graph_indices = NULL;
    uint* graph_edges = NULL;

    float* buffer_1 = NULL;
    float* buffer_2 = NULL;

    float* inv_edges_per_node = NULL;

    int num_edges = avg_edges*num_nodes;

    cudaMalloc(&graph_indices, sizeof(uint)*(1+num_nodes));
    cudaMalloc(&graph_edges, sizeof(uint)*num_edges);
    cudaMalloc(&buffer_1, sizeof(float)*num_nodes);
    cudaMalloc(&buffer_2, sizeof(float)*num_nodes);
    cudaMalloc(&inv_edges_per_node, sizeof(float)*num_nodes);
    

    // TODO: check for allocation failure
    if (!(graph_indices && graph_edges && buffer_1 && buffer_2 && inv_edges_per_node)) {
        std::cerr << "Allocation failure on GPU occurred." << std::endl;
        exit(1);
    }

    // TODO: copy data to the GPU
    cudaMemcpy(graph_indices, h_graph_indices, sizeof(uint)*(1+num_nodes), cudaMemcpyHostToDevice);
    cudaMemcpy(graph_edges, h_graph_edges, sizeof(uint)*num_edges, cudaMemcpyHostToDevice);
    cudaMemcpy(buffer_1, h_node_values_input, sizeof(float)*num_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(inv_edges_per_node, h_inv_edges_per_node, sizeof(float)*num_nodes, cudaMemcpyHostToDevice);

    start_timer(&timer);

    const int block_size = 192;

    // TODO: launch your kernels the appropriate number of iterations
    uint grid_size = std::min((uint) ((block_size+num_nodes-1)/block_size), 65535u);
    for(int i = 0; i < nr_iterations; ++i) {
        if (i % 2 == 0) {
            device_graph_propagate<<<grid_size,block_size>>>(graph_indices, graph_edges,
                                                             buffer_1, buffer_2,
                                                             inv_edges_per_node,
                                                             num_nodes);
        }
        else {
            device_graph_propagate<<<grid_size,block_size>>>(graph_indices, graph_edges,
                                                             buffer_2, buffer_1,
                                                             inv_edges_per_node,
                                                             num_nodes);
        }
    }

    check_launch("gpu graph propagate");
    double gpu_elapsed_time = stop_timer(&timer);

    // TODO: copy final data back to the host for correctness checking
    if (nr_iterations % 2 == 0)
        cudaMemcpy(h_gpu_node_values_output, buffer_1, sizeof(float)*num_nodes, cudaMemcpyDeviceToHost);
    else
        cudaMemcpy(h_gpu_node_values_output, buffer_2, sizeof(float)*num_nodes, cudaMemcpyDeviceToHost);

    // TODO: free the memory you allocated!
    cudaFree(graph_indices);
    cudaFree(graph_edges);
    cudaFree(buffer_1);
    cudaFree(buffer_2);
    cudaFree(inv_edges_per_node);

    return gpu_elapsed_time;
}
