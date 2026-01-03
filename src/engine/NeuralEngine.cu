#include "engine/NeuralEngine.h"
#include "engine/kernels/LIF_Update.cuh"
#include "engine/kernels/SpMV_Input.cuh"
#include "engine/CorticalColumnAgentic.h" // Include Agentic Header
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>

namespace neurogen {

// ============================================================================
// NEURAL ENGINE IMPLEMENTATION
// ============================================================================

NeuralEngine::NeuralEngine(int device_id) 
    : device_id_(device_id), is_initialized_(false),
      num_neurons_(0), num_synapses_(0),
      d_input_buffer_(nullptr), d_output_buffer_(nullptr) {
    
    cudaSetDevice(device_id_);
    cudaStreamCreate(&compute_stream_);
}

NeuralEngine::~NeuralEngine() {
    freeMemory();
    cudaStreamDestroy(compute_stream_);
}

bool NeuralEngine::initialize(const NetworkConfig& config) {
    cudaSetDevice(device_id_);
    
    // Check if we should use Agentic Mode (V2.5)
    // For now, we can trigger this based on a config flag or heuristic.
    // Let's assume if config.use_agentic is true (we'd need to add it to config)
    // or just default to false unless specified.
    // For this task, we'll enable it if num_neurons matches a grid size.
    
    if (config.num_neurons == 64 * 100) { // Example: 100 columns of 64 neurons
        use_agentic_mode_ = true;
        num_neurons_ = config.num_neurons; // Set this so allocateMemory works
        
        // Allocate buffers (d_input_buffer_, etc.)
        cudaMalloc(&d_input_buffer_, num_neurons_ * sizeof(float));
        cudaMalloc(&d_output_buffer_, num_neurons_ * sizeof(float));

        int cols = 10;
        int rows = 10;
        agentic_grid_.initialize(cols, rows, 64);
        is_initialized_ = true;
        std::cout << "NeuralEngine: Initialized in Agentic Mode (V2.5)" << std::endl;
        return true;
    }

    num_neurons_ = config.num_neurons;
    num_synapses_ = config.num_synapses;
    num_inputs_ = config.num_inputs;
    num_outputs_ = config.num_outputs;

    allocateMemory();
    initializeState();
    
    // Initialize Synaptic Matrix (Sparse)
    synaptic_matrix_.initialize(num_neurons_, num_neurons_, num_synapses_);
    
    // Populate with random connections (Phase 1 Baseline)
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<float> values;
    rows.reserve(num_synapses_);
    cols.reserve(num_synapses_);
    values.reserve(num_synapses_);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist_neuron(0, num_neurons_ - 1);
    std::normal_distribution<float> dist_weight(0.1f, 0.05f);

    for (size_t i = 0; i < num_synapses_; ++i) {
        rows.push_back(dist_neuron(rng)); // Post-synaptic
        cols.push_back(dist_neuron(rng)); // Pre-synaptic
        values.push_back(dist_weight(rng));
    }

    synaptic_matrix_.setFromTriplets(rows, cols, values);
    
    is_initialized_ = true;
    return true;
}

void NeuralEngine::allocateMemory() {
    // TensorNetwork
    network_state_.num_neurons = num_neurons_;
    cudaMalloc(&network_state_.d_voltage, num_neurons_ * sizeof(float));
    cudaMalloc(&network_state_.d_adaptation, num_neurons_ * sizeof(float));
    cudaMalloc(&network_state_.d_threshold, num_neurons_ * sizeof(float));
    cudaMalloc(&network_state_.d_spikes, num_neurons_ * sizeof(uint8_t));
    cudaMalloc(&network_state_.d_input_current, num_neurons_ * sizeof(float));
    cudaMalloc(&network_state_.d_last_spike_time, num_neurons_ * sizeof(float));
    
    // Buffers
    cudaMalloc(&d_input_buffer_, num_neurons_ * sizeof(float));
    cudaMalloc(&d_output_buffer_, num_neurons_ * sizeof(float));
}

void NeuralEngine::freeMemory() {
    if (use_agentic_mode_) {
        agentic_grid_.free();
        return;
    }

    if (network_state_.d_voltage) cudaFree(network_state_.d_voltage);
    if (network_state_.d_adaptation) cudaFree(network_state_.d_adaptation);
    if (network_state_.d_threshold) cudaFree(network_state_.d_threshold);
    if (network_state_.d_spikes) cudaFree(network_state_.d_spikes);
    if (network_state_.d_input_current) cudaFree(network_state_.d_input_current);
    if (network_state_.d_last_spike_time) cudaFree(network_state_.d_last_spike_time);
    
    if (d_input_buffer_) cudaFree(d_input_buffer_);
    if (d_output_buffer_) cudaFree(d_output_buffer_);
}

void NeuralEngine::initializeState() {
    // Initialize to resting potentials
    std::vector<float> initial_v(num_neurons_, -65.0f);
    cudaMemcpy(network_state_.d_voltage, initial_v.data(), num_neurons_ * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMemset(network_state_.d_adaptation, 0, num_neurons_ * sizeof(float));
    cudaMemset(network_state_.d_spikes, 0, num_neurons_ * sizeof(uint8_t));
    cudaMemset(network_state_.d_input_current, 0, num_neurons_ * sizeof(float));
}

void NeuralEngine::update(float dt, float reward_signal) {
    if (!is_initialized_) return;

    if (use_agentic_mode_) {
        // V2.5 Agentic Update
        // Note: 't' is not tracked in NeuralEngine currently, we might need to add it or pass 0.
        // For now, let's assume t is handled internally or we pass a dummy.
        static float t = 0.0f;
        agentic_grid_.update(t, dt);
        t += dt;
        return;
    }

    static float current_time = 0.0f;
    current_time += dt;

    // Phase 2 Compute Pipeline
    
    // 1. Compute Synaptic Inputs (SpMV)
    // Uses optimized kernel from kernels/SpMV_Input.cu
    // Re-uses d_output_buffer_ as temporary float spike buffer
    neurogen::kernels::computeSynapticInputs(
        synaptic_matrix_,
        network_state_.d_spikes,
        network_state_.d_input_current,
        d_output_buffer_, // Temp float spikes
        num_neurons_,
        compute_stream_
    );

    // 2. Update Neurons (LIF + kWTA)
    // Uses optimized fused kernel from kernels/LIF_Update.cu
    neurogen::kernels::LIFParams lif_params;
    lif_params.alpha = 0.95f;
    lif_params.beta = 0.90f;
    lif_params.delta = 0.5f;      // REDUCED: Less adaptation buildup (was 1.0f)
    lif_params.v_thresh = -55.0f; // LOWERED: Easier to spike (was -50.0f, needs only 10mV input now)
    lif_params.v_reset = -65.0f;
    lif_params.k_winners = 50;    // INCREASED: Allow more winners (was 25, now ~20% of 256)
    
    neurogen::kernels::launchLIFUpdate(
        network_state_.d_voltage,
        network_state_.d_adaptation,
        network_state_.d_spikes,
        network_state_.d_input_current,
        network_state_.d_last_spike_time,
        current_time,
        num_neurons_,
        dt,
        lif_params,
        compute_stream_
    );
    
    // 3. Reset Input Current for next step
    // Essential for "Emulation" vs "Simulation" - we clear accumulator
    cudaMemsetAsync(network_state_.d_input_current, 0, num_neurons_ * sizeof(float), compute_stream_);
    
    cudaStreamSynchronize(compute_stream_);
}

void NeuralEngine::processInput(const std::vector<float>& inputs) {
    if (!is_initialized_) return;
    
    if (use_agentic_mode_) {
        // In Agentic Mode, we inject current into the Input Block of columns.
        // For simplicity, map inputs 1:1 to the first N neurons of the grid.
        // A better way would be to have a dedicated input mapping.
        size_t count = std::min(inputs.size(), agentic_grid_.neuron_state.capacity);
        
        // We need to add to i_stim.
        // Since inputs is on host, we need a temp buffer or copy.
        // Let's assume d_input_buffer_ is available (we should allocate it in init even for agentic)
        if (!d_input_buffer_) {
             cudaMalloc(&d_input_buffer_, count * sizeof(float));
        }
        
        cudaMemcpy(d_input_buffer_, inputs.data(), count * sizeof(float), cudaMemcpyHostToDevice);
        
        // Add to i_stim kernel
        // We need a simple kernel to add buffer to i_stim.
        // For now, let's just copy it (overwrite) or use a lambda/kernel if we had one.
        // Since we don't have a "add_input" kernel exposed, we'll just copy for now.
        // Ideally: agentic_grid_.inject_input(d_input_buffer_, count);
        
        // Let's add inject_input to ColumnGrid
        agentic_grid_.inject_input(d_input_buffer_, count);
        return;
    }

    size_t count = std::min(inputs.size(), num_neurons_);
    // Copy directly to input current accumulator
    cudaMemcpyAsync(network_state_.d_input_current, inputs.data(), count * sizeof(float), cudaMemcpyHostToDevice, compute_stream_);
}

std::vector<float> NeuralEngine::getNeuronOutputs() {
    if (use_agentic_mode_) {
        // Return outputs from the Output Block of columns?
        // Or just all neurons?
        // Let's return all for visualization/debug.
        size_t count = agentic_grid_.neuron_state.capacity;
        std::vector<float> outputs(count);
        cudaMemcpy(outputs.data(), agentic_grid_.neuron_state.output, count * sizeof(float), cudaMemcpyDeviceToHost);
        return outputs;
    }

    std::vector<float> outputs(num_outputs_);

    // ✅ VOLTAGE-BASED READOUT (continuous signal)
    // Read membrane potentials instead of binary spikes
    std::vector<float> host_voltages(num_outputs_);
    cudaMemcpy(host_voltages.data(), network_state_.d_voltage,
               num_outputs_ * sizeof(float), cudaMemcpyDeviceToHost);

    // Convert voltage to activity signal
    // Voltage range: -65mV (rest) to ~+30mV (spike peak)
    // Map to [0, 1] range with sigmoid-like function
    for(size_t i=0; i<num_outputs_; ++i) {
        float v = host_voltages[i];

        // Shift so resting potential (-65) maps to 0
        float shifted = v + 65.0f;  // Now: 0 (rest) to 95 (peak)

        // Rectify to remove sub-threshold activity
        float rectified = std::max(0.0f, shifted);

        // Normalize to [0, 1] with sigmoid activation
        // Center around 10mV above rest (typical active neurons)
        float normalized = 1.0f / (1.0f + std::exp(-(rectified - 10.0f) * 0.1f));

        outputs[i] = normalized;
    }

    return outputs;
}

// Backward compatibility stubs
std::vector<GPUNeuronState> NeuralEngine::getNeuronStates() const {
    std::vector<float> v(num_neurons_);
    cudaMemcpy(v.data(), network_state_.d_voltage, num_neurons_ * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::vector<GPUNeuronState> states(num_neurons_);
    for(size_t i=0; i<num_neurons_; ++i) {
        states[i].V = v[i];
    }
    return states;
}

std::vector<GPUSynapse> NeuralEngine::getSynapseStates() const {
    return std::vector<GPUSynapse>(num_synapses_);
}

bool NeuralEngine::setNeuronStates(const std::vector<GPUNeuronState>& states) {
    if (states.size() != num_neurons_) return false;

    std::vector<float> v(num_neurons_);
    std::vector<float> a(num_neurons_);

    for(size_t i=0; i<num_neurons_; ++i) {
        v[i] = states[i].V;
        a[i] = states[i].u;
    }

    cudaMemcpy(network_state_.d_voltage, v.data(), num_neurons_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(network_state_.d_adaptation, a.data(), num_neurons_ * sizeof(float), cudaMemcpyHostToDevice);
    
    return true;
}

bool NeuralEngine::setSynapseStates(const std::vector<GPUSynapse>& synapses) {
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<float> weights;
    
    rows.reserve(synapses.size());
    cols.reserve(synapses.size());
    weights.reserve(synapses.size());
    
    for(const auto& s : synapses) {
        if (s.active) {
             rows.push_back(s.post_neuron_idx);
             cols.push_back(s.pre_neuron_idx);
             weights.push_back(s.weight);
        }
    }
    
    synaptic_matrix_.setFromTriplets(rows, cols, weights);
    return true;
}

} // namespace neurogen
