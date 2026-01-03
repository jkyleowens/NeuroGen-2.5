#pragma once

#include <engine/GPUNeuralStructures.h>

/**
 * @brief Kernel that accumulates per-neuron activity into compact output groups.
 *
 * @param neurons Device pointer to neuron state array.
 * @param group_sums Device pointer to per-group accumulation buffer.
 * @param group_counts Device pointer to per-group sample counts.
 * @param num_neurons Total number of neurons.
 * @param group_size Number of neurons that map to a single output group.
 * @param num_outputs Total number of output groups.
 */
__global__ void accumulateNeuronOutputsKernel(
    const GPUNeuronState* neurons,
    float* group_sums,
    int* group_counts,
    int num_neurons,
    int group_size,
    int num_outputs);

/**
 * @brief Kernel that normalizes accumulated group sums by their counts.
 *
 * @param group_sums Device pointer to per-group accumulation buffer (in/out).
 * @param group_counts Device pointer to per-group sample counts.
 * @param num_outputs Total number of output groups.
 */
__global__ void finalizeNeuronOutputsKernel(
    float* group_sums,
    const int* group_counts,
    int num_outputs);

