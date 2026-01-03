#ifndef HEBBIAN_LEARNING_KERNEL_CUH
#define HEBBIAN_LEARNING_KERNEL_CUH

#include "GPUNeuralStructures.h"

// ============================================================================
// HEBBIAN LEARNING KERNEL DECLARATIONS
// ============================================================================

/**
 * Standard Hebbian learning rule: "cells that fire together, wire together"
 */
__global__ void hebbianLearningKernel(GPUSynapse* synapses,
                                     const GPUNeuronState* neurons,
                                     float learning_rate,
                                     float dt,
                                     int num_synapses);

/**
 * Oja's learning rule: Hebbian learning with weight normalization
 */
__global__ void ojasLearningKernel(GPUSynapse* synapses,
                                  const GPUNeuronState* neurons,
                                  float learning_rate,
                                  float dt,
                                  int num_synapses);

/**
 * BCM (Bienenstock-Cooper-Munro) learning rule with sliding threshold
 */
__global__ void bcmLearningKernel(GPUSynapse* synapses,
                                 GPUNeuronState* neurons,
                                 float learning_rate,
                                 float dt,
                                 int num_synapses);

/**
 * Correlation-based learning with temporal window
 */
__global__ void correlationLearningKernel(GPUSynapse* synapses,
                                         const GPUNeuronState* neurons,
                                         float* correlation_buffer,
                                         float learning_rate,
                                         float dt,
                                         int num_synapses,
                                         int correlation_window);

#endif // HEBBIAN_LEARNING_KERNEL_CUH