/* Repeating from the tutorial, just in case you haven't looked at it.
   "kernels" or __global__ functions are the entry points to code that executes on the GPU.
   The keyword __global__ indicates to the compiler that this function is a GPU entry point.
   __global__ functions must return void, and may only be called or "launched" from code that
   executes on the CPU.
*/

typedef unsigned char uchar;
typedef unsigned int uint;

// This kernel implements a per element shift
// by naively loading one byte and shifting it
__global__ void shift_char(const uchar* input_array, uchar* output_array,
                           uchar shift_amount, uint array_length) {
   const uint tid = blockDim.x*blockIdx.x + threadIdx.x;
   const uint stride = blockDim.x*gridDim.x;
   for(uint i = tid; i < array_length; i += stride)
       output_array[i] = input_array[i] + shift_amount;
}

//Here we load 4 bytes at a time instead of just 1
//to improve the bandwidth due to a better memory
//access pattern
__global__ void shift_int(const uint* input_array, uint* output_array,
                          uint shift_amount, uint array_length) {
    const uint tid = blockDim.x*blockIdx.x + threadIdx.x;
    const uint stride = blockDim.x*gridDim.x;
    for(uint i = tid; i < array_length; i += stride)
        output_array[i] = input_array[i] + shift_amount;
}

//Here we go even further and load 8 bytes
//does it make a further improvement? Not really
__global__ void shift_int2(const uint2* input_array, uint2* output_array,
                           uint shift_amount, uint array_length) {
    const uint tid = blockDim.x*blockIdx.x + threadIdx.x;
    const uint stride = blockDim.x*gridDim.x;
    for(uint i = tid; i < array_length; i += stride) {
        output_array[i].x = input_array[i].x + shift_amount;
        output_array[i].y = input_array[i].y + shift_amount;
    }
}

//the following three kernels launch their respective kernels
//and report the time it took for the kernel to run

double doGPUShiftChar(const uchar* d_input, uchar* d_output,
                      uchar shift_amount, uint text_size, uint block_size) {
    // TODO: compute your grid dimensions
    uint grid_size = std::min(1+((text_size - 1)/block_size), 65535U);

    event_pair timer;
    start_timer(&timer);

    // TODO: launch kernel
    shift_char<<<grid_size, block_size>>>(d_input, d_output, shift_amount, text_size);

    check_launch("gpu shift cipher char");
    return stop_timer(&timer);
}

double doGPUShiftUInt(const uchar* d_input, uchar* d_output,
                      uchar shift_amount, uint text_size, uint block_size) {
    // TODO: compute your grid dimensions
    uint char_to_uint = (text_size + 3)/4;
    uint grid_size = std::min(1+((char_to_uint - 1)/block_size), 65535U);
    
    // TODO: compute 4 byte shift value - shift one byte at a time to check
    uint uint_shift = (shift_amount << 24) | (shift_amount << 16) | (shift_amount << 8) | shift_amount;

    event_pair timer;
    start_timer(&timer);

    // TODO: launch kernel
    shift_int<<<grid_size, block_size>>>((const uint*) d_input, (uint*) d_output, uint_shift, char_to_uint);


    check_launch("gpu shift cipher uint");
    return stop_timer(&timer);
}

double doGPUShiftUInt2(const uchar* d_input, uchar* d_output
                       , uchar shift_amount, uint text_size, uint block_size) {
    // TODO: compute your grid dimensions
    uint char_to_uint2 = (text_size + 7)/8;
    uint grid_size = std::min(1 + ((char_to_uint2 - 1)/block_size), 65535U);

    // TODO: compute 4 byte shift value
    uint uint2_shift = (shift_amount << 24) | (shift_amount << 16) | (shift_amount << 8) | shift_amount;

    event_pair timer;
    start_timer(&timer);

    // TODO: launch kernel
    shift_int2<<<grid_size, block_size>>>((const uint2*) d_input, (uint2*) d_output, uint2_shift, char_to_uint2);

    check_launch("gpu shift cipher uint2");
    return stop_timer(&timer);
}
