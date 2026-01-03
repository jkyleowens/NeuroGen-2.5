#ifndef NEURON_UPDATE_KERNEL_CUH
#define NEURON_UPDATE_KERNEL_CUH

#include "GPUNeuralStructures.h"

/**
 * @brief Updates the state of each neuron for one time step using the Izhikevich model.
 *
 * This kernel calculates the new membrane potential and recovery variable for each
 * neuron, and detects when a neuron fires a spike.
 *
 * @param neurons Pointer to the array of neuron states on the GPU.
 * @param current_time The current simulation time in milliseconds.
 * @param dt The simulation time step in milliseconds.
 * @param num_neurons The total number of neurons in the network.
 */
__global__ void neuronUpdateKernel(GPUNeuronState* neurons, float current_time, float dt, int num_neurons);

#endif // NEURON_UPDATE_KERNEL_CUH