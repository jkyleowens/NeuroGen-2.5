#ifndef ENHANCED_LEARNING_SYSTEM_H
#define ENHANCED_LEARNING_SYSTEM_H

#include "engine/GPUNeuralStructures.h"
#include <cuda_runtime.h>

/**
 * @brief Enhanced learning system with GPU acceleration
 * 
 * Manages multiple learning mechanisms including STDP, Hebbian learning,
 * homeostatic plasticity, and reward modulation on GPU.
 */
class EnhancedLearningSystem {
public:
    struct LearningStats {
        float total_weight_change;
        float average_trace_activity;
        float current_dopamine_level;
        float learning_progress;
        int num_active_synapses;
        float prediction_error;
        float network_activity;
        int plasticity_updates;
    };

    EnhancedLearningSystem();
    ~EnhancedLearningSystem();

    /**
     * @brief Initialize CUDA resources
     */
    bool initializeCUDA(int num_neurons, int num_synapses);

    /**
     * @brief Update learning on GPU
     */
    void updateLearningGPU(GPUSynapse* synapses, 
                          GPUNeuronState* neurons,
                          float current_time, 
                          float dt,
                          float external_reward);

    /**
     * @brief Reset episode on GPU
     */
    void resetEpisodeGPU(bool reset_traces, bool reset_rewards);

    /**
     * @brief Get learning statistics
     */
    LearningStats getStatisticsGPU() const;

    /**
     * @brief Cleanup CUDA resources
     */
    void cleanupCUDA();

private:
    bool cuda_initialized_;
    void* d_synapses_ptr_;
    void* d_neurons_ptr_;
    void* d_reward_signals_ptr_;
    void* d_trace_stats_ptr_;
    cudaStream_t learning_stream_;
    int num_neurons_;
    int num_synapses_;
    
    // Statistics tracking
    float total_weight_change_;
    float average_eligibility_trace_;
    float baseline_dopamine_;
    float learning_progress_;
    float reward_signal_;

    void update_learning(float current_time, float dt, float external_reward);
    void launch_eligibility_reset_gpu();
    void reset_eligibility_traces_gpu();
    void update_performance_metrics_gpu();
};

#endif // ENHANCED_LEARNING_SYSTEM_H

