#include "engine/CorticalColumnAgentic.h"
#include "engine/AgenticNeuronKernels.cuh"
#include "engine/AgenticSynapseKernels.cuh"
#include <cuda_runtime.h>
#include <iostream>

namespace neurogen {
namespace cortical {

// ============================================================================
// KERNELS
// ============================================================================

/**
 * @brief Computes total activity for each column (Reduction)
 */
__global__ void compute_column_activity_kernel(
    CorticalColumnAgentic* columns,
    int num_columns,
    engine::NeuronState state,
    ColumnState col_state
) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (col_idx >= num_columns) return;
    
    CorticalColumnAgentic& col = columns[col_idx];
    float activity_sum = 0.0f;
    
    // Sum output of all neurons in this column
    for (int i = 0; i < col.neuron_count; ++i) {
        int global_idx = col.get_global_idx(i);
        activity_sum += state.output[global_idx];
    }
    
    col.total_activity = activity_sum;

    // NEW: Compute Local Error and Confidence for Self-Attention
    col.compute_local_error(state, col_state);
    col.compute_confidence(2.0f); // Precision factor of 2.0
}

/**
 * @brief Applies lateral inhibition between columns
 * 
 * Logic: If a neighbor is more CONFIDENT, suppress this column's input/activity.
 * This implements "Self-Attention" where the most plausible hypothesis wins.
 */
__global__ void apply_lateral_inhibition_with_state_kernel(
    CorticalColumnAgentic* columns,
    int num_columns,
    int grid_width,
    int grid_height,
    float inhibition_strength,
    engine::NeuronState state
) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (col_idx >= num_columns) return;
    
    CorticalColumnAgentic& col = columns[col_idx];
    float my_confidence = col.confidence_score;
    float inhibition = 0.0f;
    
    int radius = 2;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int nx = col.grid_x + dx;
            int ny = col.grid_y + dy;
            if (nx >= 0 && nx < grid_width && ny >= 0 && ny < grid_height) {
                int neighbor_idx = ny * grid_width + nx;
                float neighbor_confidence = columns[neighbor_idx].confidence_score;
                
                // If neighbor is more confident, they inhibit me.
                // We use the difference in confidence to scale the inhibition.
                if (neighbor_confidence > my_confidence) {
                    inhibition += (neighbor_confidence - my_confidence);
                }
            }
        }
    }
    
    inhibition *= inhibition_strength;
    
    // Apply to Input Block
    for (int i = 0; i < col.INPUT_BLOCK_SIZE; ++i) {
        int global_idx = col.get_global_idx(col.INPUT_OFFSET + i);
        atomicAdd(&state.i_stim[global_idx], -inhibition);
    }
}

/**
 * @brief Updates Column state (Actual vs Predicted) and computes Local Error.
 * 
 * We compare the "Actual" input (from Input Block) with the "Predicted" input (from Compute Block).
 * The difference drives learning (eProp / Three-Factor Rule).
 */
__global__ void update_column_error_kernel(
    CorticalColumnAgentic* columns,
    int num_columns,
    engine::NeuronState neuron_state,
    ColumnState col_state
) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (col_idx >= num_columns) return;
    
    CorticalColumnAgentic& col = columns[col_idx];
    col.compute_local_error(neuron_state, col_state);
}

// ============================================================================
// COLUMN GRID MANAGER IMPLEMENTATION
// ============================================================================

void ColumnGrid::initialize(int w, int h, int n_per_col) {
    width = w;
    height = h;
    neurons_per_column = n_per_col;
    int num_columns = width * height;
    int total_neurons = num_columns * neurons_per_column;
    
    // 1. Initialize Global Neuron State
    neuron_state.allocate(total_neurons);

    // 1b. Initialize Column State
    column_state.allocate(num_columns);

    // 1c. Initialize Synapse State (Placeholder size for now)
    // In a real scenario, we'd load connectivity. For now, let's allocate a fixed number per neuron.
    int synapses_per_neuron = 100; 
    int total_synapses = total_neurons * synapses_per_neuron;
    synapse_state.allocate(total_synapses);
    
    // Initialize Synapse Params
    synapse_params.learning_rate = 0.01f;
    synapse_params.trace_decay = 0.95f; // exp(-dt/tau)
    synapse_params.w_max = 1.0f;
    synapse_params.w_min = -1.0f;

    // Generate Random Connectivity (Host Side)
    std::vector<float> h_weights(total_synapses);
    std::vector<int> h_pre(total_synapses);
    std::vector<int> h_post(total_synapses);
    
    // Simple random connectivity for testing
    // Each neuron connects to 'synapses_per_neuron' random other neurons
    for (int i = 0; i < total_neurons; ++i) {
        for (int s = 0; s < synapses_per_neuron; ++s) {
            int syn_idx = i * synapses_per_neuron + s;
            h_pre[syn_idx] = rand() % total_neurons; // Random pre-synaptic neuron
            h_post[syn_idx] = i;                     // This neuron is post-synaptic
            h_weights[syn_idx] = (float(rand()) / RAND_MAX) * 0.1f; // Small random weight
        }
    }
    
    cudaMemcpy(synapse_state.weights, h_weights.data(), total_synapses * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(synapse_state.pre_idx, h_pre.data(), total_synapses * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(synapse_state.post_idx, h_post.data(), total_synapses * sizeof(int), cudaMemcpyHostToDevice);
    
    // 2. Initialize Columns on Host
    columns.resize(num_columns);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            columns[idx].column_id = idx;
            columns[idx].grid_x = x;
            columns[idx].grid_y = y;
            columns[idx].neuron_start_idx = idx * neurons_per_column;
            columns[idx].neuron_count = neurons_per_column;
            columns[idx].total_activity = 0.0f;
            columns[idx].prediction_error = 0.0f;
            // global_state pointer is tricky to set here for device use, 
            // usually we pass state separately to kernels.
        }
    }
    
    // 3. Copy Columns to Device
    cudaMalloc(&d_columns, num_columns * sizeof(CorticalColumnAgentic));
    cudaMemcpy(d_columns, columns.data(), num_columns * sizeof(CorticalColumnAgentic), cudaMemcpyHostToDevice);
    
    // 4. Set Default Params
    neuron_params.tau_m = 20.0f;
    neuron_params.tau_adapt = 100.0f;
    neuron_params.v_rest = -70.0f;
    neuron_params.v_reset = -70.0f;
    neuron_params.v_thresh_base = -50.0f;
    neuron_params.adapt_inc = 0.1f;
    neuron_params.refract_time = 2.0f;
    neuron_params.output_scale = 1.0f;
    neuron_params.tau_out = 10.0f;
    
    // Initialize neurons
    engine::launch_agentic_neuron_init(total_neurons, neuron_params, neuron_state);
}

void ColumnGrid::update(float t, float dt) {
    int num_columns = width * height;
    int total_neurons = num_columns * neurons_per_column;
    
    // 0. Pre-compute decays
    neuron_params.compute_decays(dt);
    
    // 1. Update Neurons (Integrate & Fire)
    engine::launch_agentic_neuron_update(total_neurons, t, dt, neuron_params, neuron_state);
    
    // 2. Compute Column Activity & Confidence
    int threads = 256;
    int blocks = (num_columns + threads - 1) / threads;
    compute_column_activity_kernel<<<blocks, threads>>>(
        d_columns, 
        num_columns, 
        neuron_state,
        column_state // Pass column_state
    );
    
    // 3. Apply Lateral Inhibition (Self-Attention)
    apply_lateral_inhibition_with_state_kernel<<<blocks, threads>>>(
        d_columns,
        num_columns,
        width,
        height,
        0.5f, // Inhibition strength
        neuron_state
    );

    // 4. Update Column Error (Error Calculation)
    update_column_error_kernel<<<blocks, threads>>>(
        d_columns, num_columns, neuron_state, column_state
    );

    // 5. Update Synapses (Learning)
    engine::launch_synapse_update(
        synapse_state.count,
        synapse_params,
        synapse_state,
        neuron_state,
        column_state,
        d_columns,
        neurons_per_column
    );
}

__global__ void inject_input_kernel(float* i_stim, const float* input, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        i_stim[idx] += input[idx];
    }
}

void ColumnGrid::inject_input(float* d_input, int count) {
    int blockSize = 256;
    int numBlocks = (count + blockSize - 1) / blockSize;
    inject_input_kernel<<<numBlocks, blockSize>>>(neuron_state.i_stim, d_input, count);
}

void ColumnGrid::free() {
    neuron_state.free();
    column_state.free();
    synapse_state.free();
    if (d_columns) cudaFree(d_columns);
    columns.clear();
}

} // namespace cortical
} // namespace neurogen
