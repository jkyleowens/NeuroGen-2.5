#ifndef ION_CHANNEL_INITIALIZATION_CUH
#define ION_CHANNEL_INITIALIZATION_CUH

#include "GPUNeuralStructures.h"

/**
 * @brief Initializes the ion channel states and membrane potential for all neurons.
 *
 * This kernel sets each neuron to its baseline resting state before a simulation
 * begins. It ensures that variables like membrane potential, recovery state, and
 * ionic concentrations are at their correct starting values.
 *
 * @param neurons Pointer to the array of neuron states on the GPU.
 * @param num_neurons The total number of neurons in the network.
 */
__global__ void ionChannelInitializationKernel(GPUNeuronState* neurons, int num_neurons);

#endif // ION_CHANNEL_INITIALIZATION_CUH