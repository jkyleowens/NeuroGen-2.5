/**
 * @file CorticalColumnV2.h
 * @brief Cortical Column Architecture for NeuroGen 2.0
 * 
 * This file defines the cortical column structure based on the Dynamic Multimode
 * Module (D3M) model. Each column contains:
 * - 6 cortical layers (L1-L6) with biologically-inspired connectivity
 * - ALIF neurons with compartmental processing
 * - Sparse synaptic connectivity using BSR format
 * - Bidirectional generative loops for perception and imagination
 * 
 * Architecture Overview:
 * - L1: Feedback integration (apical dendrites)
 * - L2/3: Cortico-cortical communication
 * - L4: Thalamic input (feedforward)
 * - L5: Output to subcortical structures
 * - L6: Thalamic feedback
 * 
 * @version 2.0-Cortical
 * @date November 29, 2025
 */

#ifndef CORTICAL_COLUMN_V2_H
#define CORTICAL_COLUMN_V2_H

#include "ALIFNeuron.h"
#include "SparseSynapseMatrix.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <memory>
#include <string>
#include <array>
#include <vector>

namespace neurogen {
namespace cortical {

// ============================================================================
// CORTICAL LAYER DEFINITIONS
// ============================================================================

/**
 * @brief Cortical layer indices
 */
enum class CorticalLayer : int {
    L1 = 0,     // Molecular layer (apical dendrites, feedback)
    L2 = 1,     // External granular (cortico-cortical)
    L3 = 2,     // External pyramidal (cortico-cortical)
    L4 = 3,     // Internal granular (thalamic input)
    L5 = 4,     // Internal pyramidal (output)
    L6 = 5,     // Multiform (thalamic feedback)
    NUM_LAYERS = 6
};

/**
 * @brief Layer-specific parameters
 */
struct LayerParams {
    int num_excitatory;         // Number of excitatory neurons
    int num_inhibitory;         // Number of inhibitory neurons
    float exc_inh_ratio;        // E/I ratio (typically 4:1)
    float connectivity_density; // Intra-layer connectivity
    float learning_rate;        // Layer-specific learning rate
    NeuronType exc_type;        // Excitatory neuron type
    NeuronType inh_type;        // Inhibitory neuron type
};

/**
 * @brief Default layer parameters based on cortical biology
 */
inline std::array<LayerParams, 6> getDefaultLayerParams(int neurons_per_column) {
    std::array<LayerParams, 6> params;
    
    // L1: Sparse, mostly inhibitory (Martinotti cell targets)
    params[0] = {
        .num_excitatory = static_cast<int>(neurons_per_column * 0.02f),
        .num_inhibitory = static_cast<int>(neurons_per_column * 0.01f),
        .exc_inh_ratio = 2.0f,
        .connectivity_density = 0.05f,
        .learning_rate = 0.001f,
        .exc_type = NeuronType::PYRAMIDAL,
        .inh_type = NeuronType::SOM_MARTINOTTI
    };
    
    // L2/3: Dense, high connectivity (cortico-cortical)
    params[1] = {
        .num_excitatory = static_cast<int>(neurons_per_column * 0.20f),
        .num_inhibitory = static_cast<int>(neurons_per_column * 0.05f),
        .exc_inh_ratio = 4.0f,
        .connectivity_density = 0.15f,
        .learning_rate = 0.01f,
        .exc_type = NeuronType::PYRAMIDAL,
        .inh_type = NeuronType::PV_BASKET
    };
    
    params[2] = params[1]; // L3 similar to L2
    params[2].num_excitatory = static_cast<int>(neurons_per_column * 0.25f);
    params[2].num_inhibitory = static_cast<int>(neurons_per_column * 0.06f);
    
    // L4: Input layer (spiny stellate cells)
    params[3] = {
        .num_excitatory = static_cast<int>(neurons_per_column * 0.15f),
        .num_inhibitory = static_cast<int>(neurons_per_column * 0.04f),
        .exc_inh_ratio = 4.0f,
        .connectivity_density = 0.20f,
        .learning_rate = 0.005f,
        .exc_type = NeuronType::SPINY_STELLATE,
        .inh_type = NeuronType::PV_BASKET
    };
    
    // L5: Output layer (thick-tufted pyramidal)
    params[4] = {
        .num_excitatory = static_cast<int>(neurons_per_column * 0.15f),
        .num_inhibitory = static_cast<int>(neurons_per_column * 0.04f),
        .exc_inh_ratio = 4.0f,
        .connectivity_density = 0.12f,
        .learning_rate = 0.01f,
        .exc_type = NeuronType::PYRAMIDAL,
        .inh_type = NeuronType::PV_BASKET
    };
    
    // L6: Feedback layer
    params[5] = {
        .num_excitatory = static_cast<int>(neurons_per_column * 0.10f),
        .num_inhibitory = static_cast<int>(neurons_per_column * 0.03f),
        .exc_inh_ratio = 3.5f,
        .connectivity_density = 0.10f,
        .learning_rate = 0.008f,
        .exc_type = NeuronType::PYRAMIDAL,
        .inh_type = NeuronType::SOM_MARTINOTTI
    };
    
    return params;
}

// ============================================================================
// INTER-LAYER CONNECTIVITY
// ============================================================================

/**
 * @brief Canonical cortical connectivity patterns
 * Based on: Douglas & Martin (2004), Thomson & Lamy (2007)
 */
struct InterLayerConnectivity {
    // Connection probability matrix [source_layer][target_layer]
    // Values from Binzegger et al. (2004) simplified
    static constexpr float connectivity[6][6] = {
        // To:  L1    L2    L3    L4    L5    L6
        /*L1*/ {0.00f, 0.02f, 0.02f, 0.00f, 0.01f, 0.01f},
        /*L2*/ {0.05f, 0.10f, 0.15f, 0.02f, 0.08f, 0.05f},
        /*L3*/ {0.05f, 0.15f, 0.12f, 0.02f, 0.10f, 0.08f},
        /*L4*/ {0.01f, 0.20f, 0.15f, 0.08f, 0.05f, 0.10f},
        /*L5*/ {0.02f, 0.05f, 0.08f, 0.02f, 0.10f, 0.15f},
        /*L6*/ {0.02f, 0.03f, 0.05f, 0.15f, 0.08f, 0.08f}
    };
    
    // Excitatory vs inhibitory fraction of connections
    static constexpr float exc_fraction = 0.8f;
};

// ============================================================================
// CORTICAL COLUMN CLASS
// ============================================================================

/**
 * @brief Configuration for a cortical column
 */
struct CorticalColumnConfig {
    std::string name = "Column";
    int total_neurons = 10000;      // Total neurons in column
    int column_index = 0;           // Index in the cortical sheet
    
    // Neuron configuration
    bool use_compartmental = true;  // Use compartmental ALIF for pyramidal cells
    ALIFParameters alif_params;     // Base ALIF parameters
    
    // Synapse configuration
    bool use_bsr = true;            // Use Block Sparse Row format
    int bsr_block_size = 32;        // BSR block size (match warp size)
    SynapseParameters synapse_params;
    
    // Connectivity
    ConnectivityParams connectivity;
    
    // Learning
    bool enable_stdp = true;
    bool enable_homeostasis = true;
    float global_learning_rate = 0.01f;
    
    // GPU
    int gpu_device = 0;
    cudaStream_t stream = 0;
};

/**
 * @brief Cortical Column - the fundamental computational unit
 * 
 * Implements a cortical column with:
 * - 6-layer laminar structure
 * - ALIF neurons (compartmental for pyramidal cells)
 * - Sparse connectivity (CSR or BSR)
 * - STDP with eligibility traces
 * - Homeostatic plasticity
 */
class CorticalColumnV2 {
public:
    // === CONSTRUCTION ===
    CorticalColumnV2(const CorticalColumnConfig& config);
    ~CorticalColumnV2();
    
    // Prevent copying
    CorticalColumnV2(const CorticalColumnV2&) = delete;
    CorticalColumnV2& operator=(const CorticalColumnV2&) = delete;
    
    // Allow moving
    CorticalColumnV2(CorticalColumnV2&&) noexcept;
    CorticalColumnV2& operator=(CorticalColumnV2&&) noexcept;
    
    // === INITIALIZATION ===
    
    /**
     * @brief Initialize all neurons and synapses
     */
    cudaError_t initialize();
    
    /**
     * @brief Generate intra-column connectivity
     */
    cudaError_t generateConnectivity();
    
    // === SIMULATION ===
    
    /**
     * @brief Run one simulation timestep
     * @param dt Timestep in milliseconds
     * @param current_time Current simulation time
     */
    cudaError_t step(float dt, float current_time);
    
    /**
     * @brief Inject external input to L4 (feedforward)
     * @param input Input vector (size = L4 neurons)
     */
    cudaError_t injectFeedforwardInput(const float* d_input, int input_size);
    
    /**
     * @brief Inject feedback input to L1 (context/prediction)
     * @param feedback Feedback vector (size = L1 neurons)
     */
    cudaError_t injectFeedbackInput(const float* d_feedback, int feedback_size);
    
    /**
     * @brief Get output from L5 (for downstream processing)
     * @param output Pre-allocated output buffer
     */
    cudaError_t getOutput(float* d_output, int output_size) const;
    
    // === LEARNING ===
    
    /**
     * @brief Apply STDP learning rule
     * @param dopamine_signal Reward/dopamine modulation
     */
    cudaError_t applySTDP(float dopamine_signal);
    
    /**
     * @brief Apply homeostatic scaling
     */
    cudaError_t applyHomeostasis();
    
    /**
     * @brief Modulate learning with neuromodulators
     */
    cudaError_t modulateLearning(float dopamine, float acetylcholine, float serotonin);
    
    // === ACCESSORS ===
    
    const std::string& getName() const { return config_.name; }
    int getTotalNeurons() const { return total_neurons_; }
    int getNeuronsInLayer(CorticalLayer layer) const;
    int getTotalSynapses() const { return total_synapses_; }
    
    /**
     * @brief Get spike output for a layer
     */
    const float* getLayerSpikes(CorticalLayer layer) const;
    
    /**
     * @brief Get firing rates for a layer
     */
    const float* getLayerFiringRates(CorticalLayer layer) const;
    
    /**
     * @brief Get memory footprint in bytes
     */
    size_t getMemoryFootprint() const;
    
    // === STATISTICS ===
    
    struct ColumnStats {
        float mean_firing_rate;         // Hz
        float exc_firing_rate;          // Excitatory neurons
        float inh_firing_rate;          // Inhibitory neurons
        float mean_weight;              // Average synaptic weight
        float weight_std;               // Weight standard deviation
        int active_neurons;             // Neurons that spiked
        int active_synapses;            // Synapses that transmitted
        float layer_firing_rates[6];    // Per-layer firing rates
    };
    
    ColumnStats getStatistics() const;

private:
    // === INTERNAL METHODS ===
    
    cudaError_t allocateNeurons();
    cudaError_t allocateSynapses();
    cudaError_t initializeNeurons();
    cudaError_t initializeSynapses();
    cudaError_t generateConnectivityForLayer(int layer);
    
    cudaError_t updateNeurons(float dt, float current_time);
    cudaError_t propagateSynapses();
    cudaError_t updateTraces(float dt);
    
    // === MEMBER VARIABLES ===
    
    CorticalColumnConfig config_;
    
    // Neuron arrays (one per layer, or combined)
    std::array<ALIFNeuronArrays, 6> layer_neurons_;
    CompartmentalALIFArrays pyramidal_neurons_;  // For L2/3/5 pyramidal cells
    
    // Synapse matrices
    // Intra-layer connections (6 matrices, one per layer)
    std::array<CSRSynapseMatrix, 6> intra_layer_synapses_;
    
    // Inter-layer connections (6x6 possible, but only canonical ones)
    std::vector<CSRSynapseMatrix> inter_layer_synapses_;
    
    // Alternative: Single BSR matrix for all connections
    BSRSynapseMatrix<32> all_synapses_bsr_;
    
    // Neuron counts
    int total_neurons_;
    int total_synapses_;
    std::array<int, 6> neurons_per_layer_;
    std::array<int, 6> layer_offsets_;  // Start index for each layer
    
    // Layer parameters
    std::array<LayerParams, 6> layer_params_;
    
    // GPU resources
    cusparseHandle_t cusparse_handle_;
    cublasHandle_t cublas_handle_;
    
    // Temporary buffers
    float* d_temp_currents_;     // For accumulating synaptic currents
    float* d_temp_spikes_;       // For spike propagation
    
    // State tracking
    bool initialized_;
    float simulation_time_;
};

// ============================================================================
// CORTICAL SHEET (collection of columns)
// ============================================================================

/**
 * @brief A sheet of cortical columns with lateral connections
 */
class CorticalSheet {
public:
    struct Config {
        int num_columns_x = 4;          // Columns in X dimension
        int num_columns_y = 4;          // Columns in Y dimension
        int neurons_per_column = 10000; // Neurons per column
        float lateral_connectivity = 0.05f; // Inter-column connectivity
        float column_spacing_um = 500.0f;   // Physical spacing
        int gpu_device = 0;
    };
    
    CorticalSheet(const Config& config);
    ~CorticalSheet();
    
    cudaError_t initialize();
    cudaError_t step(float dt, float current_time);
    
    CorticalColumnV2* getColumn(int x, int y);
    const CorticalColumnV2* getColumn(int x, int y) const;
    
    int getNumColumns() const { return columns_.size(); }
    size_t getMemoryFootprint() const;

private:
    Config config_;
    std::vector<std::unique_ptr<CorticalColumnV2>> columns_;
    
    // Lateral (inter-column) connections
    std::vector<CSRSynapseMatrix> lateral_synapses_;
};

} // namespace cortical
} // namespace neurogen

#endif // CORTICAL_COLUMN_V2_H
