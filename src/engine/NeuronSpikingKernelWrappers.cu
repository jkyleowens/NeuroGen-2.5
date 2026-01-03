// ============================================================================
// CUDA KERNEL WRAPPER FUNCTIONS FOR NEUROSPIKINGKERNELS
// File: src/cuda/NeuronSpikingKernelWrappers.cu
// ============================================================================

#include <engine/NeuronSpikingKernels.cuh>
#include <engine/GPUNeuralStructures.h>
#include <engine/NeuronModelConstants.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>

// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================

extern "C" void launchProcessModularInteractions(GPUNeuronState* neurons, int num_neurons,
                                                int* module_assignments, float* attention_weights,
                                                float* global_inhibition, float current_time);

// ============================================================================
// EXTERNAL LINKAGE WRAPPER FUNCTIONS
// ============================================================================

/**
 * @brief Host wrapper for updateNeuronSpikes kernel with correct signature
 * 
 * This wrapper ensures proper linkage for the NetworkCUDA class while
 * maintaining the biologically accurate spike detection implementation.
 */
extern "C" void launchUpdateNeuronSpikesHost(GPUNeuronState* neurons,
                                             int num_neurons,
                                             float current_time,
                                             float dt) {
    dim3 block(256);
    dim3 grid((num_neurons + block.x - 1) / block.x);

    // Launch the actual CUDA kernel with proper parameters
    updateNeuronSpikes<<<grid, block>>>(neurons, num_neurons, current_time, dt);
    
    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error in launchUpdateNeuronSpikesHost: %s\n",
                cudaGetErrorString(error));
    }
    
    // Synchronize to ensure completion
    cudaDeviceSynchronize();
}

/**
 * @brief Host wrapper for countSpikesKernel with correct signature
 * 
 * This wrapper provides the exact function signature expected by NetworkCUDA
 * while leveraging our advanced spike counting implementation.
 */
extern "C" void launchCountSpikes(const GPUNeuronState* neurons,
                                   int* spike_count,
                                   int num_neurons,
                                   float current_time) {
    dim3 block(256);
    dim3 grid((num_neurons + block.x - 1) / block.x);

    // Launch the actual CUDA kernel
    countSpikesKernel<<<grid, block>>>(neurons, spike_count, num_neurons, current_time);
    
    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error in launchCountSpikes: %s\n",
                cudaGetErrorString(error));
    }
    
    // Synchronize to ensure completion
    cudaDeviceSynchronize();
}

/**
 * @brief Advanced wrapper for comprehensive spike processing
 * 
 * This wrapper provides a unified interface for all spike-related processing,
 * enabling the breakthrough neural architecture to handle complex spike dynamics
 * with optimal performance.
 */
extern "C" void processNeuralSpikes(GPUNeuronState* neurons, int* spike_count,
                                   float current_time,
                                   int num_neurons, float dt) {
    if (!neurons || !spike_count || num_neurons <= 0) {
        printf("Error: Invalid parameters for processNeuralSpikes\n");
        return;
    }
    
    dim3 block(256);
    dim3 grid((num_neurons + block.x - 1) / block.x);
    
    // Reset spike counter
    cudaMemset(spike_count, 0, sizeof(int));
    
    // Step 1: Update neuron spike states with biological realism
    updateNeuronSpikes<<<grid, block>>>(neurons, num_neurons, current_time, dt);
    cudaDeviceSynchronize();
    
    // Step 2: Count spikes for network statistics
    countSpikesKernel<<<grid, block>>>(neurons, spike_count, num_neurons, current_time);
    cudaDeviceSynchronize();
    
    // Check for any errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in processNeuralSpikes: %s\n", cudaGetErrorString(error));
    }
}

/**
 * @brief Wrapper for modular spike processing with attention mechanisms
 * 
 * This advanced wrapper supports the modular neural architecture by providing
 * module-aware spike processing with attention-based modulation.
 */
extern "C" void processModularSpikes(GPUNeuronState* neurons, int* spike_count,
                                    int* module_assignments, float* attention_weights,
                                    float current_time,
                                    int num_neurons, int num_modules, float dt) {
    if (!neurons || !spike_count || num_neurons <= 0) {
        return;
    }
    
    dim3 block(256);
    dim3 grid((num_neurons + block.x - 1) / block.x);
    
    // Reset spike counter
    cudaMemset(spike_count, 0, sizeof(int));
    
    // Process spikes with modular awareness
    if (module_assignments && attention_weights) {
        // Use advanced modular spike processing
        launchProcessModularInteractions(neurons, num_neurons, module_assignments,
                                       attention_weights, nullptr, current_time);
    }
    
    // Standard spike detection and counting
    updateNeuronSpikes<<<grid, block>>>(neurons, num_neurons, current_time, dt);
    countSpikesKernel<<<grid, block>>>(neurons, spike_count, num_neurons, current_time);
    
    cudaDeviceSynchronize();
}

// ============================================================================
// COMPATIBILITY LAYER FOR LEGACY INTERFACES
// ============================================================================

/**
 * @brief Legacy compatibility wrapper for older NetworkCUDA interfaces
 */
extern "C" void launchSpikeDetection(GPUNeuronState* d_neurons, int* d_spike_count,
                                    int num_neurons, float current_time) {
    processNeuralSpikes(d_neurons, d_spike_count, current_time, num_neurons, 0.1f);
}

/**
 * @brief Simplified interface for basic spike counting
 */
extern "C" int countActiveNeurons(const GPUNeuronState* neurons, int num_neurons, float current_time) {
    if (!neurons || num_neurons <= 0) return 0;
    
    int* d_count;
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));
    
    countSpikesKernel<<<(num_neurons + 255)/256, 256>>>(neurons, d_count, num_neurons, current_time);
    cudaDeviceSynchronize();
    
    int h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_count);

    return h_count;
}

// ============================================================================
// MODULAR PROCESSING FUNCTION DEFINITIONS
// ============================================================================

/**
 * @brief Process modular interactions with attention mechanisms
 * 
 * This function implements inter-modular communication and attention-based
 * modulation for the modular neural architecture.
 * NOTE: Implementation is in NeuronSpikingKernels.cu to avoid duplicate definitions
 */

// ============================================================================
// PERFORMANCE MONITORING FUNCTIONS
// ============================================================================

/**
 * @brief Monitor spike processing performance for optimization
 */
extern "C" float benchmarkSpikeProcessing(GPUNeuronState* neurons, int num_neurons,
                                         int iterations) {
    if (!neurons || num_neurons <= 0 || iterations <= 0) return 0.0f;
    
    int* d_spike_count;
    cudaMalloc(&d_spike_count, sizeof(int));
    
    // Warm up
    for (int i = 0; i < 10; i++) {
        updateNeuronSpikes<<<(num_neurons + 255)/256, 256>>>(neurons, num_neurons, 0.0f, 1.0f);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        updateNeuronSpikes<<<(num_neurons + 255)/256, 256>>>(neurons, num_neurons, 0.0f, 1.0f);
        countSpikesKernel<<<(num_neurons + 255)/256, 256>>>(neurons, d_spike_count, num_neurons, 0.0f);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    cudaFree(d_spike_count);
    
    return static_cast<float>(duration.count()) / iterations; // microseconds per iteration
}

/**
 * @brief Overloaded version with default iterations for convenience
 */
float benchmarkSpikeProcessingDefault(GPUNeuronState* neurons, int num_neurons) {
    return benchmarkSpikeProcessing(neurons, num_neurons, 100);
}

/**
 * @brief Validate spike processing accuracy for debugging
 */
extern "C" bool validateSpikeProcessing(GPUNeuronState* neurons, int num_neurons) {
    if (!neurons || num_neurons <= 0) return false;
    
    // Copy neurons to host for validation
    std::vector<GPUNeuronState> h_neurons(num_neurons);
    cudaMemcpy(h_neurons.data(), neurons, num_neurons * sizeof(GPUNeuronState), 
               cudaMemcpyDeviceToHost);
    
    // Basic validation checks
    bool valid = true;
    for (int i = 0; i < num_neurons; i++) {
        const auto& neuron = h_neurons[i];
        
        // Check for reasonable voltage values
        if (neuron.V < -100.0f || neuron.V > 100.0f) {
            printf("Warning: Neuron %d has unreasonable voltage: %f\n", i, neuron.V);
            valid = false;
        }
        
        // Check for reasonable calcium values
        for (int c = 0; c < 4; c++) {
            if (neuron.ca_conc[c] < 0.0f || neuron.ca_conc[c] > 50.0f) {
                printf("Warning: Neuron %d compartment %d has unreasonable calcium: %f\n", 
                       i, c, neuron.ca_conc[c]);
                valid = false;
            }
        }
    }
    
    return valid;
}
