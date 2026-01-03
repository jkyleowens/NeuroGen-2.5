#pragma once
#include "AgenticNeuron.h"
#include "AgenticSynapse.h"
#include <cuda_runtime.h>
#include <vector>

namespace neurogen {
namespace cortical {

// ============================================================================
// COLUMN STATE (SoA)
// ============================================================================
struct ColumnState {
    float* actual_input;    // Sum of input layer activity
    float* predicted_input; // Sum of predictive layer activity
    float* local_error;     // actual - predicted
    
    size_t total_columns;
    
    __host__ void allocate(size_t n) {
        total_columns = n;
        cudaMalloc(&actual_input, n * sizeof(float));
        cudaMalloc(&predicted_input, n * sizeof(float));
        cudaMalloc(&local_error, n * sizeof(float));
        initialize();
    }
    
    __host__ void initialize() {
        cudaMemset(actual_input, 0, total_columns * sizeof(float));
        cudaMemset(predicted_input, 0, total_columns * sizeof(float));
        cudaMemset(local_error, 0, total_columns * sizeof(float));
    }
    
    __host__ void free() {
        cudaFree(actual_input);
        cudaFree(predicted_input);
        cudaFree(local_error);
    }
};

// ============================================================================
// CORTICAL COLUMN (V2.5 Agentic)
// ============================================================================
/**
 * @brief A single Cortical Column acting as a Predictive Processing Unit.
 * 
 * Structure:
 * - Input Block (L4): Receives feedforward input
 * - Computing Block (L2/3): Hidden state / Association
 * - Prediction Block (L5/6): Output and Feedback
 * 
 * This struct manages a grid of AgenticNeurons and their local state.
 * NOTE: In V2.5, the CorticalColumnAgentic IS the minicolumn unit.
 */
struct CorticalColumnAgentic {
    // === TOPOLOGY ===
    int column_id;
    int grid_x, grid_y;
    
    // === NEURON POPULATION ===
    // Instead of separate arrays per layer, we use a single contiguous block
    // in the global AgenticNeuron arrays.
    int neuron_start_idx;
    int neuron_count;       // e.g., 64
    
    // Functional Block Offsets (relative to neuron_start_idx)
    static constexpr int INPUT_BLOCK_SIZE = 16;
    static constexpr int COMPUTE_BLOCK_SIZE = 32;
    static constexpr int OUTPUT_BLOCK_SIZE = 16;
    
    static constexpr int INPUT_OFFSET = 0;
    static constexpr int COMPUTE_OFFSET = 16;
    static constexpr int OUTPUT_OFFSET = 48;
    
    // === STATE ===
    float total_activity;   // Sum of outputs (for lateral inhibition)
    float prediction_error; // Difference between input and prediction
    float predicted_activity; // Stored prediction strength for confidence calc
    float confidence_score; // Prediction strength weighted by precision (low error)
    
    // === DEVICE POINTERS (for lateral inhibition kernel) ===
    // These point to the global arrays but are stored here for convenience
    // when launching column-specific kernels.
    engine::NeuronState* global_state; 
    
    // === METHODS ===
    
    /**
     * @brief Get the global index of a neuron within this column
     */
    __host__ __device__ int get_global_idx(int local_idx) const {
        return neuron_start_idx + local_idx;
    }
    
    /**
     * @brief Check if this column inhibits a neighbor
     */
    __host__ __device__ bool inhibits(const CorticalColumnAgentic& neighbor) const {
        // Simple radius-based inhibition
        int dx = grid_x - neighbor.grid_x;
        int dy = grid_y - neighbor.grid_y;
        return (dx*dx + dy*dy) <= 25; // Radius 5
    }

    /**
     * @brief Computes the error for this column (minicolumn).
     * 
     * 1. Computes Actual vs Predicted input.
     * 2. Computes the Local Error.
     */
    __device__ void compute_local_error(
        engine::NeuronState& state,
        ColumnState& col_state
    ) {
        float actual = 0.0f;
        float predicted = 0.0f;
        
        // 1. Sum Actual Input (from Input Block neurons)
        for (int i = 0; i < INPUT_BLOCK_SIZE; ++i) {
            int global_idx = get_global_idx(INPUT_OFFSET + i);
            actual += state.output[global_idx];
        }
        
        // 2. Sum Predicted Input (from Compute Block neurons)
        for (int i = 0; i < COMPUTE_BLOCK_SIZE; ++i) {
            int global_idx = get_global_idx(COMPUTE_OFFSET + i);
            predicted += state.output[global_idx];
        }
        
        // 3. Compute & Store Error
        float error = actual - predicted;
        
        col_state.actual_input[column_id] = actual;
        col_state.predicted_input[column_id] = predicted;
        col_state.local_error[column_id] = error;
        
        // Store for internal use
        prediction_error = error;
        predicted_activity = predicted;
    }

    /**
     * @brief Computes the confidence score for self-attention.
     * 
     * Confidence = Prediction_Strength - ( |Error| * Precision_Factor )
     * 
     * A column is confident if it predicts strongly AND has low error.
     * If error is high (surprise), confidence drops, allowing other columns to win.
     */
    __device__ void compute_confidence(float precision_factor = 2.0f) {
        // Confidence = Prediction - ( |Error| * Precision )
        // If Error is 0, Confidence = Prediction (Pure confidence)
        // If Error is high, Confidence becomes negative (Suppressed)
        
        float abs_error = fabsf(prediction_error);
        confidence_score = predicted_activity - (abs_error * precision_factor);
    }
};

// ============================================================================
// COLUMN GRID MANAGER
// ============================================================================
/**
 * @brief Manages the grid of cortical columns
 */
class ColumnGrid {
public:
    std::vector<CorticalColumnAgentic> columns;
    CorticalColumnAgentic* d_columns; // GPU copy of column structs
    
    int width;
    int height;
    int neurons_per_column;
    
    // Global Neuron State (SoA)
    engine::NeuronState neuron_state;
    engine::NeuronParams neuron_params;

    // Column State (SoA)
    ColumnState column_state;

    // Synapse State (SoA)
    engine::SynapseState synapse_state;
    engine::SynapseParams synapse_params;
    
    void initialize(int w, int h, int n_per_col);
    void update(float t, float dt);
    void inject_input(float* d_input, int count);
    void free();
};

} // namespace cortical
} // namespace neurogen
