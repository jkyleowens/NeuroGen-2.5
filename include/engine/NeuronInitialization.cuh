#ifndef NEURON_INITIALIZATION_CUH
#define NEURON_INITIALIZATION_CUH

#include "GPUNeuralStructures.h"

/**
 * @brief Initializes the state of all neurons on the GPU.
 *
 * This kernel sets each neuron to its baseline resting state before a simulation
 * begins. It is essential for ensuring a correct and reproducible starting point.
 * It configures membrane potential, recovery variables, spike times, and all
 * homeostatic and neuromodulatory parameters to their default initial values.
 *
 * @param neurons Pointer to the array of neuron states on the GPU.
 * @param num_neurons The total number of neurons in the network.
 */
__global__ void neuronInitializationKernel(GPUNeuronState* neurons, int num_neurons);

#endif // NEURON_INITIALIZATION_CUH