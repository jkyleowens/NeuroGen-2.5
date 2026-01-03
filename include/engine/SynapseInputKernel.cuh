#ifndef SYNAPSE_INPUT_KERNEL_CUH
#define SYNAPSE_INPUT_KERNEL_CUH

#include <cuda_runtime.h>

// Forward declarations
struct GPUSynapse;
struct GPUNeuronState;

/**
 * CUDA kernel for processing synaptic inputs
 * @param synapses Array of synapses
 * @param neurons Array of neuron states
 * @param num_synapses Number of synapses to process
 */
__global__ void synapseInputKernel(GPUSynapse* synapses, GPUNeuronState* neurons, int num_synapses);

#endif // SYNAPSE_INPUT_KERNEL_CUH