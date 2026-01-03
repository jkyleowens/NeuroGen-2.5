#ifndef NEURON_SPIKING_KERNELS_CUH
#define NEURON_SPIKING_KERNELS_CUH

#include <cuda_runtime.h>

// Forward declarations
struct GPUNeuronState;
struct GPUSpikeEvent;

/**
 * CUDA kernel to count the number of neurons that have spiked
 * @param neurons Array of neuron states
 * @param spike_count Pointer to counter for number of spikes
 * @param num_neurons Total number of neurons
 */

 /**
 * @brief Resets the `spiked` flag for all neurons at the start of a simulation step.
 */
__global__ void resetSpikeFlags(GPUNeuronState* neurons, int num_neurons);

__global__ void countSpikesKernel(const GPUNeuronState* neurons,
                                 int* spike_count, int num_neurons,
                                 float current_time);

/**
 * CUDA kernel to update the spiking state of neurons
 * @param neurons Array of neuron states
 * @param threshold Voltage threshold for spike
 * @param num_neurons Total number of neurons
 */
__global__ void updateNeuronSpikes(GPUNeuronState* neurons,
                                  int num_neurons, float current_time,
                                  float dt);

/**
 * CUDA kernel to detect spikes and record spike events
 * @param neurons Array of neuron states
 * @param spikes Array to store spike events
 * @param threshold Voltage threshold for spike
 * @param spike_count Pointer to counter for number of spikes
 * @param num_neurons Total number of neurons
 * @param current_time Current simulation time
 */
__global__ void detectSpikes(const GPUNeuronState* neurons,
                            GPUSpikeEvent* spikes,
                            int* spike_count, int num_neurons,
                            float current_time,
                            int* module_assignments,
                            int max_spike_events);

#endif // NEURON_SPIKING_KERNELS_CUH
