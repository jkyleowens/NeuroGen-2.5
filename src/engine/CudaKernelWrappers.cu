#include <cuda_runtime.h>
#include <cstdio>
#include <engine/GPUNeuralStructures.h>
#include <engine/EnhancedSTDPKernel.cuh>
#include <engine/HebbianLearningKernel.cuh>
#include <engine/HomeostaticMechanismsKernel.cuh>
#include <engine/NeuromodulationKernels.cuh>

extern "C" {

// ============================================================================
// MISSING WRAPPER FUNCTIONS FOR ENHANCED LEARNING SYSTEM
// ============================================================================

void launch_eligibility_reset_wrapper(void* d_synapses, int num_synapses) {
    // Reset eligibility traces to zero
    cudaError_t error = cudaMemset(d_synapses, 0, num_synapses * sizeof(float) * 4); // Assuming 4 floats per eligibility state
    if (error != cudaSuccess) {
        printf("CUDA Error in eligibility reset: %s\n", cudaGetErrorString(error));
    }
}

void launch_eligibility_update_wrapper(void* d_synapses, const void* d_neurons,
                                      float current_time, float dt, int num_synapses) {
    // This would update eligibility traces - for now, use enhanced STDP as proxy
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    enhancedSTDPKernel<<<grid, block>>>(
        static_cast<GPUSynapse*>(d_synapses),
        static_cast<const GPUNeuronState*>(d_neurons),
        current_time, dt, num_synapses
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in eligibility update: %s\n", cudaGetErrorString(error));
    }
}

void launch_trace_monitoring_wrapper(const void* d_synapses, int num_synapses, void* d_trace_stats) {
    // Simple implementation - copy some synapse statistics
    // This would normally compute trace statistics on GPU
    float stats[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // sum, max, mean, variance
    cudaError_t error = cudaMemcpy(d_trace_stats, stats, 4 * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("CUDA Error in trace monitoring: %s\n", cudaGetErrorString(error));
    }
}

void launch_reward_modulation_wrapper(void* d_synapses, void* d_neurons, float reward,
                                     float current_time, float dt, int num_synapses) {
    // Apply reward modulation to synapses - simplified implementation
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    // Use enhanced STDP kernel as a proxy for reward modulation
    enhancedSTDPKernel<<<grid, block>>>(
        static_cast<GPUSynapse*>(d_synapses),
        static_cast<const GPUNeuronState*>(d_neurons),
        current_time, dt, num_synapses
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in reward modulation: %s\n", cudaGetErrorString(error));
    }
}

void launch_correlation_learning_wrapper(void* d_synapses, const void* d_neurons,
                                        void* d_correlation_matrix, float learning_rate,
                                        float dt, int num_synapses, int matrix_size) {
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    correlationLearningKernel<<<grid, block>>>(
        static_cast<GPUSynapse*>(d_synapses),
        static_cast<const GPUNeuronState*>(d_neurons),
        static_cast<float*>(d_correlation_matrix),
        learning_rate, dt, num_synapses, matrix_size
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in correlation learning: %s\n", cudaGetErrorString(error));
    }
}

void launch_reward_prediction_error_wrapper(const void* d_actual_reward,
                                           void* d_predicted_rewards, int num_timesteps) {
    // Simple reward prediction error computation
    // This would normally be a more complex kernel
    cudaError_t error = cudaMemcpy(d_predicted_rewards, d_actual_reward, 
                                  num_timesteps * sizeof(float), cudaMemcpyDeviceToDevice);
    if (error != cudaSuccess) {
        printf("CUDA Error in reward prediction error: %s\n", cudaGetErrorString(error));
    }
}

// ============================================================================
// ENHANCED STDP KERNEL WRAPPER
// ============================================================================
void launch_enhanced_stdp_wrapper(void* d_synapses, const void* d_neurons,
                                 float current_time, float dt, int num_synapses) {
    
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    // Call the actual kernel with correct signature
    enhancedSTDPKernel<<<grid, block>>>(
        static_cast<GPUSynapse*>(d_synapses),
        static_cast<const GPUNeuronState*>(d_neurons),
        current_time, dt, num_synapses
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in enhanced STDP kernel: %s\n", cudaGetErrorString(error));
    }
}

// ============================================================================
// HEBBIAN LEARNING KERNEL WRAPPERS
// ============================================================================
void launch_hebbian_learning_wrapper(void* d_synapses, const void* d_neurons,
                                    float current_time, float dt, int num_synapses) {
    
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    hebbianLearningKernel<<<grid, block>>>(
        static_cast<GPUSynapse*>(d_synapses),
        static_cast<const GPUNeuronState*>(d_neurons),
        current_time, dt, num_synapses
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in Hebbian learning kernel: %s\n", cudaGetErrorString(error));
    }
}

void launch_bcm_learning_wrapper(void* d_synapses, void* d_neurons,
                                float learning_rate, float dt, int num_synapses) {
    
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    bcmLearningKernel<<<grid, block>>>(
        static_cast<GPUSynapse*>(d_synapses),
        static_cast<GPUNeuronState*>(d_neurons),
        learning_rate, dt, num_synapses
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in BCM learning kernel: %s\n", cudaGetErrorString(error));
    }
}

void launch_ojas_learning_wrapper(void* d_synapses, const void* d_neurons,
                                 float learning_rate, float dt, int num_synapses) {
    
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    ojasLearningKernel<<<grid, block>>>(
        static_cast<GPUSynapse*>(d_synapses),
        static_cast<const GPUNeuronState*>(d_neurons),
        learning_rate, dt, num_synapses
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in Oja's learning kernel: %s\n", cudaGetErrorString(error));
    }
}

// ============================================================================
// HOMEOSTATIC MECHANISMS KERNEL WRAPPERS
// ============================================================================
void launch_synaptic_scaling_wrapper(void* d_neurons, void* d_synapses,
                                    float current_time, float dt,
                                    int num_synapses, int num_neurons) {
    
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    synapticScalingKernel<<<grid, block>>>(
        static_cast<GPUNeuronState*>(d_neurons),
        static_cast<GPUSynapse*>(d_synapses),
        num_neurons, num_synapses, current_time
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in synaptic scaling kernel: %s\n", cudaGetErrorString(error));
    }
}

void launch_apply_synaptic_scaling_wrapper(void* d_synapses, const void* d_neurons,
                                          int num_synapses) {
    
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    applySynapticScalingKernel<<<grid, block>>>(
        static_cast<GPUSynapse*>(d_synapses),
        static_cast<const GPUNeuronState*>(d_neurons),
        num_synapses
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in apply synaptic scaling kernel: %s\n", cudaGetErrorString(error));
    }
}

void launch_weight_normalization_wrapper(void* d_synapses, void* d_synapse_counts,
                                        int num_synapses, int num_neurons) {
    
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    weightNormalizationKernel<<<grid, block>>>(
        static_cast<GPUSynapse*>(d_synapses),
        static_cast<int*>(d_synapse_counts),
        num_synapses, num_neurons
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in weight normalization kernel: %s\n", cudaGetErrorString(error));
    }
}

void launch_activity_regulation_wrapper(void* d_neurons, float current_time,
                                       float dt, int num_neurons) {
    
    dim3 block(256);
    dim3 grid((num_neurons + block.x - 1) / block.x);
    
    activityRegulationKernel<<<grid, block>>>(
        static_cast<GPUNeuronState*>(d_neurons),
        current_time, dt, num_neurons
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in activity regulation kernel: %s\n", cudaGetErrorString(error));
    }
}

void launch_intrinsic_plasticity_wrapper(void* d_neurons, int num_neurons) {
    
    dim3 block(256);
    dim3 grid((num_neurons + block.x - 1) / block.x);
    
    intrinsicPlasticityKernel<<<grid, block>>>(
        static_cast<GPUNeuronState*>(d_neurons),
        num_neurons
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in intrinsic plasticity kernel: %s\n", cudaGetErrorString(error));
    }
}

// ============================================================================
// NEUROMODULATION KERNEL WRAPPERS
// ============================================================================
void launch_intrinsic_neuromodulation_wrapper(void* d_neurons, float ach_level,
                                              float ser_level, int num_neurons) {
    
    dim3 block(256);
    dim3 grid((num_neurons + block.x - 1) / block.x);
    
    applyIntrinsicNeuromodulationKernel<<<grid, block>>>(
        static_cast<GPUNeuronState*>(d_neurons),
        ach_level, ser_level, num_neurons
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in intrinsic neuromodulation kernel: %s\n", cudaGetErrorString(error));
    }
}

void launch_synaptic_neuromodulation_wrapper(void* d_synapses, float ach_level,
                                            int num_synapses) {
    
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    applySynapticNeuromodulationKernel<<<grid, block>>>(
        static_cast<GPUSynapse*>(d_synapses),
        ach_level, num_synapses
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in synaptic neuromodulation kernel: %s\n", cudaGetErrorString(error));
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================
void cuda_memory_copy_wrapper(void* dst, const void* src, size_t size, int direction) {
    cudaMemcpyKind kind = (direction == 0) ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
    cudaError_t error = cudaMemcpy(dst, src, size, kind);
    if (error != cudaSuccess) {
        printf("CUDA Memory Copy Error: %s\n", cudaGetErrorString(error));
    }
}

int cuda_check_last_error_wrapper(void) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
        return 1;
    }
    return 0;
}

void cuda_device_synchronize_wrapper(void) {
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        printf("CUDA Device Synchronize Error: %s\n", cudaGetErrorString(error));
    }
}

void cuda_malloc_wrapper(void** ptr, size_t size) {
    cudaError_t error = cudaMalloc(ptr, size);
    if (error != cudaSuccess) {
        printf("CUDA Malloc Error: %s\n", cudaGetErrorString(error));
        *ptr = nullptr;
    }
}

void cuda_free_wrapper(void* ptr) {
    if (ptr != nullptr) {
        cudaError_t error = cudaFree(ptr);
        if (error != cudaSuccess) {
            printf("CUDA Free Error: %s\n", cudaGetErrorString(error));
        }
    }
}

} // extern "C"
