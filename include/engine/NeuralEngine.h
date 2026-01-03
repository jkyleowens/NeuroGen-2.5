#pragma once
#include "engine/TensorNetwork.h"
#include "engine/SparseMatrix.h"
#include "engine/NetworkConfig.h"
#include "engine/GPUNeuralStructures.h" // For legacy structs if needed
#include "engine/CorticalColumnAgentic.h" // V2.5 Agentic Architecture
#include <vector>
#include <memory>

namespace neurogen {

/**
 * @brief High-Performance Neural Engine (Phase 1)
 * 
 * Replaces NetworkCUDA. Orchestrates the linear algebra operations
 * for the biological neural network emulation.
 */
class NeuralEngine {
public:
    explicit NeuralEngine(int device_id);
    ~NeuralEngine();

    /**
     * @brief Initialize the engine with network configuration
     */
    bool initialize(const NetworkConfig& config);

    /**
     * @brief Execute one simulation step
     * @param dt Time step in ms
     * @param reward_signal Global dopamine/reward signal
     */
    void update(float dt, float reward_signal);

    /**
     * @brief Process external inputs (current injection)
     */
    void processInput(const std::vector<float>& inputs);

    /**
     * @brief Retrieve voltage-based activity of output neurons
     * Returns continuous signal (0-1) based on membrane potential
     */
    std::vector<float> getNeuronOutputs();

    // === Persistence / Debug Accessors ===
    // These provide backward compatibility or snapshot capabilities
    
    /**
     * @brief Get legacy AoS state (slow, for snapshots/debug only)
     */
    std::vector<GPUNeuronState> getNeuronStates() const;
    
    /**
     * @brief Get legacy Synapse state (slow, for snapshots/debug only)
     */
    std::vector<GPUSynapse> getSynapseStates() const;

    /**
     * @brief Set neuron state from legacy AoS (slow, for loading checkpoints)
     */
    bool setNeuronStates(const std::vector<GPUNeuronState>& states);

    /**
     * @brief Set synapse state from legacy AoS (slow, rebuilds matrix)
     */
    bool setSynapseStates(const std::vector<GPUSynapse>& synapses);

private:
    int device_id_;
    bool is_initialized_;
    
    size_t num_neurons_;
    size_t num_synapses_;
    size_t num_inputs_;
    size_t num_outputs_;

    // === Core Data Structures ===
    TensorNetwork network_state_;
    SparseMatrix synaptic_matrix_;

    // === V2.5 Agentic Architecture ===
    cortical::ColumnGrid agentic_grid_;
    bool use_agentic_mode_ = false;

    // === Buffers ===
    float* d_input_buffer_;     // Buffer for external inputs
    float* d_output_buffer_;    // Buffer for reading back outputs

    // === CUDA Streams ===
    cudaStream_t compute_stream_;

    // === Private Helpers ===
    void allocateMemory();
    void freeMemory();
    void initializeState();
    void launchUpdateKernel(float dt, float reward);
};

} // namespace neurogen
