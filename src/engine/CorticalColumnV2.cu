/**
 * @file CorticalColumnV2.cu
 * @brief Implementation of Cortical Column Architecture
 *
 * This file implements the CorticalColumnV2 class, providing:
 * - 6-layer cortical column with biological connectivity
 * - ALIF neurons with compartmental processing
 * - Sparse synaptic propagation (CSR/BSR)
 * - STDP learning with eligibility traces
 * - Homeostatic plasticity
 * - Biology-First: Dale's Law and topographic connectivity
 *
 * @version 2.0-Biology-First
 * @date December 30, 2025
 */

#include "engine/CorticalColumnV2.h"
#include "engine/BioConstants.h"
#include "engine/ALIFKernels.cuh"
#include "engine/SparseKernels.cuh"
#include "engine/STDPKernels.cuh"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>

// Forward declare BioConnectivityGenerator
namespace neurogen {
    class BioConnectivityGenerator;

    // External function for biological wiring
    void generateCorticalWiring(
        std::vector<int>& row_ptr,
        std::vector<int>& col_ind,
        std::vector<float>& values,
        int num_neurons,
        int width, int height
    );
}

namespace neurogen {
namespace cortical {

// ============================================================================
// CORTICAL COLUMN IMPLEMENTATION
// ============================================================================

CorticalColumnV2::CorticalColumnV2(const CorticalColumnConfig& config)
    : config_(config)
    , total_neurons_(0)
    , total_synapses_(0)
    , cusparse_handle_(nullptr)
    , cublas_handle_(nullptr)
    , d_temp_currents_(nullptr)
    , d_temp_spikes_(nullptr)
    , initialized_(false)
    , simulation_time_(0.0f)
{
    // Initialize layer parameters
    layer_params_ = getDefaultLayerParams(config_.total_neurons);
    
    // Calculate neuron counts per layer
    neurons_per_layer_.fill(0);
    layer_offsets_.fill(0);
    
    int offset = 0;
    for (int i = 0; i < 6; ++i) {
        neurons_per_layer_[i] = layer_params_[i].num_excitatory + layer_params_[i].num_inhibitory;
        layer_offsets_[i] = offset;
        offset += neurons_per_layer_[i];
    }
    total_neurons_ = offset;
}

CorticalColumnV2::~CorticalColumnV2() {
    // Free neuron arrays
    for (auto& layer : layer_neurons_) {
        layer.free();
    }
    
    // Free pyramidal neurons
    pyramidal_neurons_.free();
    
    // Free synapse matrices
    for (auto& syn : intra_layer_synapses_) {
        syn.free();
    }
    for (auto& syn : inter_layer_synapses_) {
        syn.free();
    }
    all_synapses_bsr_.free();
    
    // Free temporary buffers
    if (d_temp_currents_) cudaFree(d_temp_currents_);
    if (d_temp_spikes_) cudaFree(d_temp_spikes_);
    
    // Destroy cuSPARSE/cuBLAS handles
    if (cusparse_handle_) cusparseDestroy(cusparse_handle_);
    if (cublas_handle_) cublasDestroy(cublas_handle_);
}

CorticalColumnV2::CorticalColumnV2(CorticalColumnV2&& other) noexcept
    : config_(std::move(other.config_))
    , layer_neurons_(std::move(other.layer_neurons_))
    , pyramidal_neurons_(std::move(other.pyramidal_neurons_))
    , intra_layer_synapses_(std::move(other.intra_layer_synapses_))
    , inter_layer_synapses_(std::move(other.inter_layer_synapses_))
    , all_synapses_bsr_(std::move(other.all_synapses_bsr_))
    , total_neurons_(other.total_neurons_)
    , total_synapses_(other.total_synapses_)
    , neurons_per_layer_(other.neurons_per_layer_)
    , layer_offsets_(other.layer_offsets_)
    , layer_params_(other.layer_params_)
    , cusparse_handle_(other.cusparse_handle_)
    , cublas_handle_(other.cublas_handle_)
    , d_temp_currents_(other.d_temp_currents_)
    , d_temp_spikes_(other.d_temp_spikes_)
    , initialized_(other.initialized_)
    , simulation_time_(other.simulation_time_)
{
    // Null out moved-from object
    other.cusparse_handle_ = nullptr;
    other.cublas_handle_ = nullptr;
    other.d_temp_currents_ = nullptr;
    other.d_temp_spikes_ = nullptr;
    other.initialized_ = false;
}

CorticalColumnV2& CorticalColumnV2::operator=(CorticalColumnV2&& other) noexcept {
    if (this != &other) {
        // Clean up existing resources
        for (auto& layer : layer_neurons_) layer.free();
        pyramidal_neurons_.free();
        for (auto& syn : intra_layer_synapses_) syn.free();
        for (auto& syn : inter_layer_synapses_) syn.free();
        all_synapses_bsr_.free();
        if (d_temp_currents_) cudaFree(d_temp_currents_);
        if (d_temp_spikes_) cudaFree(d_temp_spikes_);
        if (cusparse_handle_) cusparseDestroy(cusparse_handle_);
        if (cublas_handle_) cublasDestroy(cublas_handle_);
        
        // Move from other
        config_ = std::move(other.config_);
        layer_neurons_ = std::move(other.layer_neurons_);
        pyramidal_neurons_ = std::move(other.pyramidal_neurons_);
        intra_layer_synapses_ = std::move(other.intra_layer_synapses_);
        inter_layer_synapses_ = std::move(other.inter_layer_synapses_);
        all_synapses_bsr_ = std::move(other.all_synapses_bsr_);
        total_neurons_ = other.total_neurons_;
        total_synapses_ = other.total_synapses_;
        neurons_per_layer_ = other.neurons_per_layer_;
        layer_offsets_ = other.layer_offsets_;
        layer_params_ = other.layer_params_;
        cusparse_handle_ = other.cusparse_handle_;
        cublas_handle_ = other.cublas_handle_;
        d_temp_currents_ = other.d_temp_currents_;
        d_temp_spikes_ = other.d_temp_spikes_;
        initialized_ = other.initialized_;
        simulation_time_ = other.simulation_time_;
        
        other.cusparse_handle_ = nullptr;
        other.cublas_handle_ = nullptr;
        other.d_temp_currents_ = nullptr;
        other.d_temp_spikes_ = nullptr;
        other.initialized_ = false;
    }
    return *this;
}

// ============================================================================
// INITIALIZATION
// ============================================================================

cudaError_t CorticalColumnV2::initialize() {
    if (initialized_) {
        return cudaSuccess;
    }
    
    cudaError_t err;
    
    // Set GPU device
    err = cudaSetDevice(config_.gpu_device);
    if (err != cudaSuccess) return err;
    
    // Create cuSPARSE handle
    cusparseStatus_t spErr = cusparseCreate(&cusparse_handle_);
    if (spErr != CUSPARSE_STATUS_SUCCESS) {
        return cudaErrorUnknown;
    }
    
    // Create cuBLAS handle
    cublasStatus_t blErr = cublasCreate(&cublas_handle_);
    if (blErr != CUBLAS_STATUS_SUCCESS) {
        return cudaErrorUnknown;
    }
    
    // Allocate neurons
    err = allocateNeurons();
    if (err != cudaSuccess) return err;
    
    // Initialize neurons
    err = initializeNeurons();
    if (err != cudaSuccess) return err;
    
    // Allocate temporary buffers
    err = cudaMalloc(&d_temp_currents_, total_neurons_ * sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&d_temp_spikes_, total_neurons_ * sizeof(float));
    if (err != cudaSuccess) return err;
    
    initialized_ = true;
    return cudaSuccess;
}

cudaError_t CorticalColumnV2::allocateNeurons() {
    cudaError_t err;
    
    for (int i = 0; i < 6; ++i) {
        int n = neurons_per_layer_[i];
        layer_neurons_[i].params = config_.alif_params;
        err = layer_neurons_[i].allocate(n);
        if (err != cudaSuccess) return err;
    }
    
    // Allocate compartmental neurons for pyramidal cells (L2/3/5)
    if (config_.use_compartmental) {
        int num_pyramidal = layer_params_[1].num_excitatory 
                         + layer_params_[2].num_excitatory 
                         + layer_params_[4].num_excitatory;
        pyramidal_neurons_.params = config_.alif_params;
        err = pyramidal_neurons_.allocate(num_pyramidal);
        if (err != cudaSuccess) return err;
    }
    
    return cudaSuccess;
}

cudaError_t CorticalColumnV2::initializeNeurons() {
    cudaError_t err;
    
    for (int i = 0; i < 6; ++i) {
        err = kernels::launchALIFInit(layer_neurons_[i], config_.stream);
        if (err != cudaSuccess) return err;
    }
    
    return cudaSuccess;
}

cudaError_t CorticalColumnV2::generateConnectivity() {
    cudaError_t err;
    
    // Estimate synapse counts for each layer pair
    // Based on canonical cortical connectivity matrix
    
    // Intra-layer connections
    for (int layer = 0; layer < 6; ++layer) {
        int n = neurons_per_layer_[layer];
        float density = layer_params_[layer].connectivity_density;
        int estimated_synapses = static_cast<int>(n * n * density);
        
        err = intra_layer_synapses_[layer].allocate(n, n, estimated_synapses);
        if (err != cudaSuccess) return err;
        
        // Generate random connectivity on device
        // (In production, this would use the connectivity generation kernels)
        err = generateConnectivityForLayer(layer);
        if (err != cudaSuccess) return err;
    }
    
    // Inter-layer connections based on canonical connectivity
    for (int src = 0; src < 6; ++src) {
        for (int dst = 0; dst < 6; ++dst) {
            float conn_prob = InterLayerConnectivity::connectivity[src][dst];
            if (conn_prob > 0.001f) {  // Only create if meaningful connectivity
                CSRSynapseMatrix matrix;
                
                int n_src = neurons_per_layer_[src];
                int n_dst = neurons_per_layer_[dst];
                int estimated = static_cast<int>(n_src * n_dst * conn_prob);
                
                if (estimated > 0) {
                    err = matrix.allocate(n_src, n_dst, estimated);
                    if (err != cudaSuccess) return err;
                    
                    inter_layer_synapses_.push_back(std::move(matrix));
                }
            }
        }
    }
    
    // Count total synapses
    total_synapses_ = 0;
    for (const auto& syn : intra_layer_synapses_) {
        total_synapses_ += syn.nnz;
    }
    for (const auto& syn : inter_layer_synapses_) {
        total_synapses_ += syn.nnz;
    }
    
    return cudaSuccess;
}

cudaError_t CorticalColumnV2::generateConnectivityForLayer(int layer) {
    // BIOLOGY-FIRST: Use topographic connectivity with Dale's Law
    // This replaces random connectivity with biologically-realistic patterns

    CSRSynapseMatrix& syn = intra_layer_synapses_[layer];
    int n = neurons_per_layer_[layer];

    // Calculate layer dimensions for topographic mapping
    // Approximate square grid for simplicity
    int width = static_cast<int>(std::sqrt(static_cast<float>(n)));
    int height = (n + width - 1) / width;

    std::cout << "🧬 Generating biological connectivity for layer " << layer
              << " (" << n << " neurons, " << width << "x" << height << " grid)" << std::endl;

    // Use BioConnectivityGenerator for topographic wiring with Dale's Law
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<float> weights;

    // Generate biological connectivity
    try {
        neurogen::generateCorticalWiring(row_ptr, col_idx, weights, n, width, height);
    } catch (...) {
        std::cerr << "⚠️  BioConnectivityGenerator not available, using fallback" << std::endl;

        // Fallback to original random connectivity if biological generator fails
        float density = layer_params_[layer].connectivity_density;
        row_ptr.clear();
        col_idx.clear();
        weights.clear();

        row_ptr.push_back(0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i != j && (float)rand() / RAND_MAX < density) {
                    col_idx.push_back(j);
                    float w = 0.1f + 0.02f * ((float)rand() / RAND_MAX - 0.5f);
                    weights.push_back(std::max(0.0f, std::min(1.0f, w)));
                }
            }
            row_ptr[i + 1] = col_idx.size();
        }
    }

    // Reallocate to actual size
    syn.nnz = col_idx.size();

    // Upload to GPU
    cudaError_t err;
    err = cudaMemcpy(syn.d_row_ptr, row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;

    err = cudaMemcpy(syn.d_col_idx, col_idx.data(), syn.nnz * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;

    err = cudaMemcpy(syn.d_weights, weights.data(), syn.nnz * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;

    // Initialize eligibility traces to zero
    err = cudaMemset(syn.d_eligibility, 0, syn.nnz * sizeof(float));
    if (err != cudaSuccess) return err;

    err = cudaMemset(syn.d_pre_trace, 0, syn.num_pre * sizeof(float));
    if (err != cudaSuccess) return err;

    err = cudaMemset(syn.d_post_trace, 0, syn.num_post * sizeof(float));

    return err;
}

// ============================================================================
// SIMULATION
// ============================================================================

cudaError_t CorticalColumnV2::step(float dt, float current_time) {
    if (!initialized_) {
        return cudaErrorNotReady;
    }
    
    cudaError_t err;
    
    // 1. Reset input currents
    for (int layer = 0; layer < 6; ++layer) {
        err = kernels::launchResetCurrents(
            layer_neurons_[layer].d_current,
            neurons_per_layer_[layer],
            config_.stream
        );
        if (err != cudaSuccess) return err;
    }
    
    // 2. Propagate synapses (intra-layer)
    err = propagateSynapses();
    if (err != cudaSuccess) return err;
    
    // 3. Update neuron dynamics
    err = updateNeurons(dt, current_time);
    if (err != cudaSuccess) return err;
    
    // 4. Update STDP traces
    err = updateTraces(dt);
    if (err != cudaSuccess) return err;
    
    simulation_time_ = current_time;
    
    return cudaSuccess;
}

cudaError_t CorticalColumnV2::updateNeurons(float dt, float current_time) {
    cudaError_t err;
    
    for (int layer = 0; layer < 6; ++layer) {
        err = kernels::launchALIFUpdate(
            layer_neurons_[layer],
            dt,
            current_time,
            config_.stream
        );
        if (err != cudaSuccess) return err;
    }
    
    return cudaSuccess;
}

cudaError_t CorticalColumnV2::propagateSynapses() {
    cudaError_t err;
    
    // Intra-layer propagation
    for (int layer = 0; layer < 6; ++layer) {
        err = kernels::launchCSRSpMV(
            intra_layer_synapses_[layer],
            layer_neurons_[layer].d_spike_output,
            layer_neurons_[layer].d_current,
            1.0f,
            config_.stream
        );
        if (err != cudaSuccess) return err;
    }
    
    // Inter-layer propagation
    // Map inter-layer synapses to source/destination layers
    int syn_idx = 0;
    for (int src = 0; src < 6; ++src) {
        for (int dst = 0; dst < 6; ++dst) {
            float conn_prob = InterLayerConnectivity::connectivity[src][dst];
            if (conn_prob > 0.001f && syn_idx < inter_layer_synapses_.size()) {
                err = kernels::launchCSRSpMV(
                    inter_layer_synapses_[syn_idx],
                    layer_neurons_[src].d_spike_output,
                    layer_neurons_[dst].d_current,
                    1.0f,
                    config_.stream
                );
                if (err != cudaSuccess) return err;
                syn_idx++;
            }
        }
    }
    
    return cudaSuccess;
}

cudaError_t CorticalColumnV2::updateTraces(float dt) {
    cudaError_t err;
    
    float tau_trace = config_.synapse_params.stdp_tau_pre;
    
    for (int layer = 0; layer < 6; ++layer) {
        // Update presynaptic traces
        err = kernels::launchUpdatePreTrace(
            intra_layer_synapses_[layer].d_pre_trace,
            layer_neurons_[layer].d_spike_output,
            tau_trace,
            dt,
            neurons_per_layer_[layer],
            config_.stream
        );
        if (err != cudaSuccess) return err;
        
        // Update postsynaptic traces
        err = kernels::launchUpdatePostTrace(
            intra_layer_synapses_[layer].d_post_trace,
            layer_neurons_[layer].d_spike_output,
            tau_trace,
            dt,
            neurons_per_layer_[layer],
            config_.stream
        );
        if (err != cudaSuccess) return err;
    }
    
    return cudaSuccess;
}

// ============================================================================
// INPUT/OUTPUT
// ============================================================================

cudaError_t CorticalColumnV2::injectFeedforwardInput(const float* d_input, int input_size) {
    // Inject into L4 (feedforward input layer)
    int l4_neurons = neurons_per_layer_[static_cast<int>(CorticalLayer::L4)];
    int copy_size = std::min(input_size, l4_neurons);
    
    // Add to existing currents
    // Simple version: copy directly (in production, use kernel to add)
    return cudaMemcpy(
        layer_neurons_[static_cast<int>(CorticalLayer::L4)].d_current,
        d_input,
        copy_size * sizeof(float),
        cudaMemcpyDeviceToDevice
    );
}

cudaError_t CorticalColumnV2::injectFeedbackInput(const float* d_feedback, int feedback_size) {
    // Inject into L1 (feedback/context layer)
    int l1_neurons = neurons_per_layer_[static_cast<int>(CorticalLayer::L1)];
    int copy_size = std::min(feedback_size, l1_neurons);
    
    return cudaMemcpy(
        layer_neurons_[static_cast<int>(CorticalLayer::L1)].d_current,
        d_feedback,
        copy_size * sizeof(float),
        cudaMemcpyDeviceToDevice
    );
}

cudaError_t CorticalColumnV2::getOutput(float* d_output, int output_size) const {
    // Get output from L5 (output layer)
    int l5_neurons = neurons_per_layer_[static_cast<int>(CorticalLayer::L5)];
    int copy_size = std::min(output_size, l5_neurons);
    
    // Copy firing rates as output
    return cudaMemcpy(
        d_output,
        layer_neurons_[static_cast<int>(CorticalLayer::L5)].d_firing_rate,
        copy_size * sizeof(float),
        cudaMemcpyDeviceToDevice
    );
}

// ============================================================================
// LEARNING
// ============================================================================

cudaError_t CorticalColumnV2::applySTDP(float dopamine_signal) {
    if (!config_.enable_stdp) {
        return cudaSuccess;
    }
    
    cudaError_t err;
    
    kernels::STDPParams stdp_params;
    stdp_params.tau_pre = config_.synapse_params.stdp_tau_pre;
    stdp_params.tau_post = config_.synapse_params.stdp_tau_post;
    stdp_params.A_plus = config_.synapse_params.stdp_a_plus;
    stdp_params.A_minus = config_.synapse_params.stdp_a_minus;
    stdp_params.learning_rate = config_.global_learning_rate;
    
    // Apply eligibility-trace STDP with reward modulation
    for (int layer = 0; layer < 6; ++layer) {
        // Update eligibility traces
        err = kernels::launchEligibilityUpdate(
            intra_layer_synapses_[layer],
            layer_neurons_[layer].d_spike_output,
            layer_neurons_[layer].d_spike_output,
            stdp_params,
            1.0f,  // dt
            config_.stream
        );
        if (err != cudaSuccess) return err;
        
        // Apply reward-modulated weight changes
        err = kernels::launchRewardModulation(
            intra_layer_synapses_[layer],
            dopamine_signal,
            layer_params_[layer].learning_rate,
            config_.stream
        );
        if (err != cudaSuccess) return err;
    }
    
    return cudaSuccess;
}

cudaError_t CorticalColumnV2::applyHomeostasis() {
    if (!config_.enable_homeostasis) {
        return cudaSuccess;
    }
    
    cudaError_t err;
    
    float target_rate = 5.0f;  // Target 5 Hz
    float scaling_rate = 0.001f;
    
    for (int layer = 0; layer < 6; ++layer) {
        err = kernels::launchHomeostaticScaling(
            intra_layer_synapses_[layer],
            layer_neurons_[layer].d_firing_rate,
            target_rate,
            scaling_rate,
            config_.stream
        );
        if (err != cudaSuccess) return err;
    }
    
    return cudaSuccess;
}

cudaError_t CorticalColumnV2::modulateLearning(float dopamine, float acetylcholine, float serotonin) {
    cudaError_t err;
    
    for (int layer = 0; layer < 6; ++layer) {
        err = kernels::launchNeuromodulatedSTDP(
            intra_layer_synapses_[layer],
            dopamine,
            acetylcholine,
            serotonin,
            layer_params_[layer].learning_rate,
            config_.stream
        );
        if (err != cudaSuccess) return err;
    }
    
    return cudaSuccess;
}

// ============================================================================
// ACCESSORS
// ============================================================================

int CorticalColumnV2::getNeuronsInLayer(CorticalLayer layer) const {
    return neurons_per_layer_[static_cast<int>(layer)];
}

const float* CorticalColumnV2::getLayerSpikes(CorticalLayer layer) const {
    return layer_neurons_[static_cast<int>(layer)].d_spike_output;
}

const float* CorticalColumnV2::getLayerFiringRates(CorticalLayer layer) const {
    return layer_neurons_[static_cast<int>(layer)].d_firing_rate;
}

size_t CorticalColumnV2::getMemoryFootprint() const {
    size_t total = 0;
    
    // Neurons
    for (const auto& layer : layer_neurons_) {
        total += layer.getMemoryFootprint();
    }
    total += pyramidal_neurons_.getMemoryFootprint();
    
    // Synapses
    for (const auto& syn : intra_layer_synapses_) {
        total += syn.getMemoryFootprint();
    }
    for (const auto& syn : inter_layer_synapses_) {
        total += syn.getMemoryFootprint();
    }
    
    // Temporary buffers
    total += total_neurons_ * sizeof(float) * 2;
    
    return total;
}

CorticalColumnV2::ColumnStats CorticalColumnV2::getStatistics() const {
    ColumnStats stats = {};
    
    // This would normally copy data from GPU and compute statistics
    // For now, return placeholder values
    stats.mean_firing_rate = 0.0f;
    stats.exc_firing_rate = 0.0f;
    stats.inh_firing_rate = 0.0f;
    stats.mean_weight = 0.1f;
    stats.weight_std = 0.02f;
    stats.active_neurons = 0;
    stats.active_synapses = 0;
    
    for (int i = 0; i < 6; ++i) {
        stats.layer_firing_rates[i] = 0.0f;
    }
    
    return stats;
}

// ============================================================================
// CORTICAL SHEET IMPLEMENTATION
// ============================================================================

CorticalSheet::CorticalSheet(const Config& config)
    : config_(config)
{
}

CorticalSheet::~CorticalSheet() {
    // Columns are unique_ptrs, will be cleaned up automatically
}

cudaError_t CorticalSheet::initialize() {
    cudaError_t err;
    
    // Create columns
    int total_columns = config_.num_columns_x * config_.num_columns_y;
    columns_.reserve(total_columns);
    
    for (int y = 0; y < config_.num_columns_y; ++y) {
        for (int x = 0; x < config_.num_columns_x; ++x) {
            CorticalColumnConfig col_config;
            col_config.name = "Column_" + std::to_string(x) + "_" + std::to_string(y);
            col_config.total_neurons = config_.neurons_per_column;
            col_config.column_index = y * config_.num_columns_x + x;
            col_config.gpu_device = config_.gpu_device;
            
            auto column = std::make_unique<CorticalColumnV2>(col_config);
            err = column->initialize();
            if (err != cudaSuccess) return err;
            
            err = column->generateConnectivity();
            if (err != cudaSuccess) return err;
            
            columns_.push_back(std::move(column));
        }
    }
    
    // Generate lateral (inter-column) connections
    // ... (implement if needed)
    
    return cudaSuccess;
}

cudaError_t CorticalSheet::step(float dt, float current_time) {
    cudaError_t err;
    
    // Update all columns
    for (auto& column : columns_) {
        err = column->step(dt, current_time);
        if (err != cudaSuccess) return err;
    }
    
    // Propagate lateral connections
    // ... (implement if needed)
    
    return cudaSuccess;
}

CorticalColumnV2* CorticalSheet::getColumn(int x, int y) {
    if (x < 0 || x >= config_.num_columns_x || y < 0 || y >= config_.num_columns_y) {
        return nullptr;
    }
    return columns_[y * config_.num_columns_x + x].get();
}

const CorticalColumnV2* CorticalSheet::getColumn(int x, int y) const {
    if (x < 0 || x >= config_.num_columns_x || y < 0 || y >= config_.num_columns_y) {
        return nullptr;
    }
    return columns_[y * config_.num_columns_x + x].get();
}

size_t CorticalSheet::getMemoryFootprint() const {
    size_t total = 0;
    for (const auto& column : columns_) {
        total += column->getMemoryFootprint();
    }
    for (const auto& syn : lateral_synapses_) {
        total += syn.getMemoryFootprint();
    }
    return total;
}

} // namespace cortical
} // namespace neurogen
