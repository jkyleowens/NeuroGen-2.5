#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace neurogen {
namespace engine {

// ============================================================================
// SYNAPSE PARAMETERS
// ============================================================================
struct SynapseParams {
    float learning_rate;    // Global learning rate
    float trace_decay;      // Decay factor for eligibility trace (exp(-dt/tau))
    float w_max;            // Maximum weight
    float w_min;            // Minimum weight
    
    __host__ void compute_decays(float dt, float tau_trace) {
        trace_decay = expf(-dt / tau_trace);
    }
};

// ============================================================================
// SYNAPSE STATE (SoA)
// ============================================================================
struct SynapseState {
    float* weights;         // Synaptic weight
    float* eligibility;     // Eligibility trace (e_ji)
    int* pre_idx;           // Index of pre-synaptic neuron
    int* post_idx;          // Index of post-synaptic neuron
    
    size_t count;           // Total number of synapses
    
    __host__ void allocate(size_t n) {
        count = n;
        cudaMalloc(&weights, n * sizeof(float));
        cudaMalloc(&eligibility, n * sizeof(float));
        cudaMalloc(&pre_idx, n * sizeof(int));
        cudaMalloc(&post_idx, n * sizeof(int));
        initialize();
    }
    
    __host__ void initialize() {
        cudaMemset(weights, 0, count * sizeof(float));
        cudaMemset(eligibility, 0, count * sizeof(float));
        cudaMemset(pre_idx, 0, count * sizeof(int));
        cudaMemset(post_idx, 0, count * sizeof(int));
    }
    
    __host__ void upload(const std::vector<float>& h_weights, 
                         const std::vector<int>& h_pre, 
                         const std::vector<int>& h_post) {
        if (h_weights.size() > count) {
            // Error or reallocate
            return; 
        }
        cudaMemcpy(weights, h_weights.data(), h_weights.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(pre_idx, h_pre.data(), h_pre.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(post_idx, h_post.data(), h_post.size() * sizeof(int), cudaMemcpyHostToDevice);
        // Reset eligibility
        cudaMemset(eligibility, 0, count * sizeof(float));
    }

    __host__ void free() {
        cudaFree(weights);
        cudaFree(eligibility);
        cudaFree(pre_idx);
        cudaFree(post_idx);
    }
};

} // namespace engine
} // namespace neurogen
