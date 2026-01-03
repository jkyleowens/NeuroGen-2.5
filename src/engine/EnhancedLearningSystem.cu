// ============================================================================
// ENHANCED LEARNING SYSTEM CUDA IMPLEMENTATION
// File: src/cuda/EnhancedLearningSystem.cu
// ============================================================================

#include <engine/EnhancedLearningSystem.h>
#include <engine/GPUNeuralStructures.h>
#include <engine/EnhancedSTDPKernel.cuh>
#include <engine/HebbianLearningKernel.cuh>
#include <engine/HomeostaticMechanismsKernel.cuh>
#include <engine/EligibilityAndRewardKernels.cuh>
#include <engine/RewardModulationKernel.cuh>
#include <engine/NeuromodulationKernels.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// ============================================================================
// CUDA IMPLEMENTATION OF EnhancedLearningSystem GPU METHODS
// These methods handle GPU-specific learning operations
// ============================================================================

void EnhancedLearningSystem::updateLearningGPU(GPUSynapse* synapses, 
                                              GPUNeuronState* neurons,
                                              float current_time, 
                                              float dt,
                                              float external_reward) {
    
    if (!cuda_initialized_) {
        return;
    }
    
    // Store device pointers (cast to void* for header compatibility)
    d_synapses_ptr_ = static_cast<void*>(synapses);
    d_neurons_ptr_ = static_cast<void*>(neurons);
    
    // Update main learning mechanisms
    update_learning(current_time, dt, external_reward);
    
    // Synchronize GPU execution
    cudaStreamSynchronize(learning_stream_);
}

void EnhancedLearningSystem::resetEpisodeGPU(bool reset_traces, bool reset_rewards) {
    if (!cuda_initialized_) {
        return;
    }
    
    if (reset_traces) {
        // Reset eligibility traces
        launch_eligibility_reset_gpu();
    }
    
    if (reset_rewards) {
        // Reset reward-related variables
        float zero_values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (d_reward_signals_ptr_) {
            cudaMemcpy(d_reward_signals_ptr_, zero_values, 4 * sizeof(float), cudaMemcpyHostToDevice);
        }
    }
    
    cudaStreamSynchronize(learning_stream_);
}

EnhancedLearningSystem::LearningStats EnhancedLearningSystem::getStatisticsGPU() const {
    LearningStats stats;
    
    // Initialize with current tracked values
    stats.total_weight_change = total_weight_change_;
    stats.average_trace_activity = average_eligibility_trace_;
    stats.current_dopamine_level = baseline_dopamine_;
    stats.prediction_error = 0.0f;
    stats.network_activity = 0.0f;
    stats.plasticity_updates = 0;
    
    // Get network activity from GPU if available
    if (cuda_initialized_ && d_trace_stats_ptr_) {
        float trace_stats[4] = {0};
        cudaMemcpy(trace_stats, d_trace_stats_ptr_, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        stats.network_activity = trace_stats[0];
        stats.average_trace_activity = trace_stats[1];
    }
    
    return stats;
}

// ============================================================================
// PRIVATE CUDA HELPER METHODS
// ============================================================================

void EnhancedLearningSystem::launch_eligibility_reset_gpu() {
    if (!cuda_initialized_ || !d_synapses_ptr_) {
        return;
    }
    
    // Launch eligibility reset kernel
    GPUSynapse* synapses = static_cast<GPUSynapse*>(d_synapses_ptr_);
    
    dim3 blockSize(256);
    dim3 gridSize((num_synapses_ + blockSize.x - 1) / blockSize.x);
    
    // Reset eligibility traces to zero
    cudaMemset(synapses, 0, num_synapses_ * sizeof(float)); // Reset eligibility field
    
    cudaStreamSynchronize(learning_stream_);
}

void EnhancedLearningSystem::reset_eligibility_traces_gpu() {
    if (!cuda_initialized_ || !d_trace_stats_ptr_) {
        return;
    }
    
    // Reset eligibility traces using memset (simple approach)
    size_t synapse_memory_size = num_synapses_ * sizeof(float) * 4; // Estimate for eligibility traces
    cudaMemset(d_synapses_ptr_, 0, synapse_memory_size);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error in eligibility reset: %s\n", cudaGetErrorString(error));
    }
}

// ============================================================================
// MAIN LEARNING UPDATE
// ============================================================================

void EnhancedLearningSystem::update_learning(float current_time, float dt, float external_reward) {
    if (!cuda_initialized_) {
        return;
    }
    
    // Update eligibility traces
    launch_eligibility_reset_gpu();
    
    // Apply reward modulation to synapses
    if (d_reward_signals_ptr_ && d_synapses_ptr_) {
        // Simple reward application - in practice would use CUDA kernels
        // For now, just track the reward
        reward_signal_ = external_reward;
    }
    
    // Update performance metrics
    update_performance_metrics_gpu();
}

// ============================================================================
// CUDA KERNEL LAUNCH HELPERS
// ============================================================================

void EnhancedLearningSystem::update_performance_metrics_gpu() {
    if (!cuda_initialized_ || !d_trace_stats_ptr_) {
        return;
    }
    
    // Copy trace statistics from GPU
    float trace_stats[4];
    cudaMemcpy(trace_stats, d_trace_stats_ptr_, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Update performance metrics
    average_eligibility_trace_ = trace_stats[0] / std::max(1, num_synapses_);
    
    // Update learning progress based on weight changes
    float progress = std::min(1.0f, total_weight_change_ / (num_synapses_ * 0.1f));
    learning_progress_ = progress;
    
    // Update total weight change
    total_weight_change_ += trace_stats[2]; // Assuming trace_stats[2] contains weight change magnitude
}

// ============================================================================
// CUDA ERROR CHECKING
// ============================================================================

void checkCudaErrors() {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in Enhanced Learning System: %s\n", 
               cudaGetErrorString(error));
    }
}
