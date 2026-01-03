#ifndef HOMEOSTATIC_MECHANISMS_KERNEL_CUH
#define HOMEOSTATIC_MECHANISMS_KERNEL_CUH

#include "GPUNeuralStructures.h"

#include <cuda_runtime.h>

/**
 * @brief Adjusts incoming synaptic weights to maintain a target firing rate.
 */
__global__ void synapticScalingKernel(GPUNeuronState* neurons,
                                     GPUSynapse* synapses,
                                     int num_neurons,
                                     int total_synapses,
                                     float current_time);

/**
 * @brief Apply synaptic scaling factors
 */
__global__ void applySynapticScalingKernel(GPUSynapse* synapses,
                                          const GPUNeuronState* neurons,
                                          int num_synapses);

/**
 * @brief Weight normalization kernel
 */
__global__ void weightNormalizationKernel(GPUSynapse* synapses,
                                         int* synapse_counts,
                                         int num_synapses,
                                         int num_neurons);

/**
 * @brief Activity regulation kernel
 */
__global__ void activityRegulationKernel(GPUNeuronState* neurons,
                                        float current_time,
                                        float dt,
                                        int num_neurons);

/**
 * @brief Network homeostatic monitoring
 */
__global__ void networkHomeostaticMonitoringKernel(const GPUNeuronState* neurons,
                                                  const GPUSynapse* synapses,
                                                  float* network_stats,
                                                  int num_neurons,
                                                  int num_synapses);

/**
 * @brief Adjusts a neuron's intrinsic excitability to maintain a target activity level.
 */
__global__ void intrinsicPlasticityKernel(GPUNeuronState* neurons, int num_neurons);

#endif // HOMEOSTATIC_MECHANISMS_KERNEL_CUH