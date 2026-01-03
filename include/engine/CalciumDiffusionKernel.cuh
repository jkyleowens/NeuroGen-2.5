#ifndef CALCIUM_DIFFUSION_KERNEL_CUH
#define CALCIUM_DIFFUSION_KERNEL_CUH

#include "GPUNeuralStructures.h"

/**
 * @brief Updates the calcium concentration in each neuron's dendritic compartments.
 *
 * This kernel models the influx and decay of calcium, which is a key factor
 * in determining synaptic plasticity (LTP/LTD).
 *
 * @param neurons Pointer to the array of neuron states on the GPU.
 * @param current_time The current simulation time in milliseconds.
 * @param dt The simulation time step in milliseconds.
 * @param num_neurons The total number of neurons in the network.
 */
__global__ void calciumDiffusionKernel(GPUNeuronState* neurons, float current_time, float dt, int num_neurons);

#endif // CALCIUM_DIFFUSION_KERNEL_CUH