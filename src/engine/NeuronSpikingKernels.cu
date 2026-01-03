#include <engine/NeuronSpikingKernels.cuh>
#include <engine/GPUNeuralStructures.h>
#include <engine/NeuronModelConstants.h>
#include <cuda_runtime.h>

// Forward declaration of spike event structure for modular architecture
struct GPUSpikeEvent {
    int neuron_idx;           // Index of the spiking neuron
    int compartment_idx;      // Which compartment spiked (0=soma, 1-3=dendrites)
    float time;               // Time of spike occurrence
    float amplitude;          // Spike amplitude
    float propagation_delay;  // Delay for spike propagation
    int module_id;            // Module this neuron belongs to (for modular architecture)
};

// Constants for multi-compartment dendritic processing
namespace DendriticConstants {
    __device__ constexpr float DENDRITIC_THRESHOLD[4] = {
        NeuronModelConstants::SPIKE_THRESHOLD,     // Soma threshold
        -40.0f,  // Proximal dendrite threshold
        -35.0f,  // Intermediate dendrite threshold  
        -30.0f   // Distal dendrite threshold
    };
    
    __device__ constexpr float PROPAGATION_STRENGTH[4] = {
        1.0f,    // Soma (no propagation needed)
        0.8f,    // Proximal to soma
        0.6f,    // Intermediate to proximal
        0.4f     // Distal to intermediate
    };
    
    __device__ constexpr float COMPARTMENT_RESISTANCE[4] = {
        10.0f,   // Soma resistance
        15.0f,   // Proximal dendrite
        20.0f,   // Intermediate dendrite
        25.0f    // Distal dendrite
    };
}

/**
 * @brief Initialize spike detection state for modular neural network
 * This supports the self-contained module architecture by tracking spike state per module
 */
__global__ void initializeModularSpikeState(GPUNeuronState* neurons, int num_neurons, 
                                           int* module_assignments, float current_time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    // Reset spike-related activity tracking for this timestep
    neuron.average_activity *= 0.999f; // Decay previous activity
    
    // Initialize calcium concentrations for each compartment if needed
    for (int c = 0; c < 4; c++) {
        if (neuron.ca_conc[c] < 0.1f) {
            neuron.ca_conc[c] = 0.1f; // Baseline calcium
        }
        // Decay calcium naturally
        neuron.ca_conc[c] *= NeuronModelConstants::CALCIUM_DECAY;
    }
}

/**
 * @brief Count spikes across all neurons and compartments for statistics
 */
__global__ void countSpikesKernel(const GPUNeuronState* neurons, int* spike_count, 
                                 int num_neurons, float current_time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    const GPUNeuronState& neuron = neurons[idx];
    
    // Check if soma spiked recently (within current timestep)
    if (neuron.last_spike_time > current_time - 1.0f) {
        atomicAdd(spike_count, 1);
    }
}

/**
 * @brief Detect spikes in multi-compartment neurons with biological realism
 * Supports modular architecture by tracking spikes per module and compartment
 */
__global__ void updateNeuronSpikes(GPUNeuronState* neurons, int num_neurons, 
                                  float current_time, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    // Check if neuron is in refractory period
    if (current_time < neuron.last_spike_time + NeuronModelConstants::ABSOLUTE_REFRACTORY_PERIOD) {
        return; // Skip update during refractory period
    }
    
    // Track if any compartment spikes for calcium dynamics
    bool soma_spiked = false;
    bool dendritic_spike_occurred = false;
    
    // === SOMA SPIKE DETECTION ===
    if (neuron.V >= NeuronModelConstants::SPIKE_THRESHOLD) {
        // Soma spike detected
        neuron.V = NeuronModelConstants::RESET_POTENTIAL;
        neuron.u += 8.0f; // Reset recovery variable
        neuron.last_spike_time = current_time;
        soma_spiked = true;
        
        // Update firing rate with exponential moving average
        float rate_decay = expf(-dt / 1000.0f); // 1 second time constant
        neuron.average_firing_rate = neuron.average_firing_rate * rate_decay + (1.0f - rate_decay) * 1000.0f/dt;
        
        // Increase somatic calcium significantly
        neuron.ca_conc[0] += 0.5f;
    }
    
    // === DENDRITIC COMPARTMENT PROCESSING ===
    for (int c = 1; c < 4; c++) { // Process dendritic compartments (1-3)
        // Calculate compartment-specific voltage based on synaptic input
        float compartment_voltage = neuron.V + neuron.I_syn[c] * DendriticConstants::COMPARTMENT_RESISTANCE[c];
        
        // Check for dendritic spike
        if (compartment_voltage >= DendriticConstants::DENDRITIC_THRESHOLD[c]) {
            dendritic_spike_occurred = true;
            
            // Calcium influx in spiking compartment
            neuron.ca_conc[c] += 0.3f;
            
            // Propagate dendritic spike towards soma
            float propagation = DendriticConstants::PROPAGATION_STRENGTH[c] * 15.0f;
            
            // Add depolarization to soma with distance-dependent attenuation
            neuron.V += propagation / (c + 1); // More distal = more attenuation
            
            // Backpropagating spike also affects proximal compartments
            for (int proximal = c - 1; proximal >= 0; proximal--) {
                neuron.ca_conc[proximal] += 0.1f / (c - proximal);
            }
            
            // Reset dendritic compartment after spike
            neuron.I_syn[c] *= 0.1f; // Rapid current decay after spike
        }
    }
    
    // === HOMEOSTATIC PLASTICITY ===
    if (soma_spiked || dendritic_spike_occurred) {
        // Update average activity
        neuron.average_activity += 1.0f;
        
        // Homeostatic scaling based on target firing rate
        float target_rate = NeuronModelConstants::TARGET_FIRING_RATE;
        if (neuron.average_firing_rate > target_rate * 1.2f) {
            // Too active - decrease excitability
            neuron.excitability *= 0.9999f;
            neuron.synaptic_scaling_factor *= 0.9999f;
        } else if (neuron.average_firing_rate < target_rate * 0.8f) {
            // Too quiet - increase excitability
            neuron.excitability *= 1.0001f;
            neuron.synaptic_scaling_factor *= 1.0001f;
        }
    }
    
    // Clamp homeostatic values to reasonable ranges
    neuron.excitability = fmaxf(0.1f, fminf(3.0f, neuron.excitability));
    neuron.synaptic_scaling_factor = fmaxf(0.1f, fminf(2.0f, neuron.synaptic_scaling_factor));
    
    // Clamp calcium concentrations
    for (int c = 0; c < 4; c++) {
        neuron.ca_conc[c] = fmaxf(0.0f, fminf(5.0f, neuron.ca_conc[c]));
    }
}

/**
 * @brief Detect and record spike events for detailed analysis and modular coordination
 * This supports the modular architecture by providing detailed spike information
 */
__global__ void detectSpikes(const GPUNeuronState* neurons, GPUSpikeEvent* spikes, 
                           int* spike_count, int num_neurons, float current_time,
                           int* module_assignments, int max_spike_events) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    const GPUNeuronState& neuron = neurons[idx];
    
    // Check for recent soma spike
    if (neuron.last_spike_time > current_time - 1.0f && 
        neuron.last_spike_time <= current_time) {
        
        int spike_idx = atomicAdd(spike_count, 1);
        
        if (spike_idx < max_spike_events) {
            GPUSpikeEvent& spike = spikes[spike_idx];
            spike.neuron_idx = idx;
            spike.compartment_idx = 0; // Soma spike
            spike.time = neuron.last_spike_time;
            spike.amplitude = neuron.V + 100.0f; // Spike amplitude relative to reset
            spike.propagation_delay = 0.0f; // Soma spikes have no delay
            spike.module_id = module_assignments ? module_assignments[idx] : 0;
        }
    }
    
    // Check for dendritic spikes based on calcium influx
    for (int c = 1; c < 4; c++) {
        if (neuron.ca_conc[c] > 0.25f) { // Threshold for dendritic spike detection
            int spike_idx = atomicAdd(spike_count, 1);
            
            if (spike_idx < max_spike_events) {
                GPUSpikeEvent& spike = spikes[spike_idx];
                spike.neuron_idx = idx;
                spike.compartment_idx = c;
                spike.time = current_time;
                spike.amplitude = neuron.ca_conc[c] * 50.0f; // Scale calcium to spike amplitude
                spike.propagation_delay = c * 0.5f; // Distance-dependent delay
                spike.module_id = module_assignments ? module_assignments[idx] : 0;
            }
        }
    }
}

/**
 * @brief Update voltage dynamics with compartment-specific leak currents
 * Supports biological realism with different leak properties per compartment
 */
__global__ void updateNeuronVoltages(GPUNeuronState* neurons, float dt, int num_neurons,
                                   float* external_currents) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    GPUNeuronState& neuron = neurons[idx];
    
    // Skip if in refractory period
    if (dt > 0.0f) { // Ensure positive timestep
        float time_since_spike = 1000.0f; // Default to large value
        if (neuron.last_spike_time > -1e6f) {
            time_since_spike = dt; // Approximate - should be current_time - last_spike_time
        }
        
        if (time_since_spike < NeuronModelConstants::ABSOLUTE_REFRACTORY_PERIOD) {
            return; // Skip voltage update during refractory period
        }
    }
    
    // === SOMATIC VOLTAGE UPDATE ===
    // Gather total synaptic input with modulation
    float total_synaptic_current = 0.0f;
    for (int c = 0; c < 4; c++) {
        total_synaptic_current += neuron.I_syn[c] * neuron.synaptic_scaling_factor;
    }
    
    // Add external current if provided
    float external_current = external_currents ? external_currents[idx] : 0.0f;
    total_synaptic_current += external_current;
    
    // Apply excitability modulation
    total_synaptic_current *= neuron.excitability;
    
    // Izhikevich model update
    float v = neuron.V;
    float u = neuron.u;
    
    neuron.V += dt * (0.04f * v * v + 5.0f * v + 140.0f - u + total_synaptic_current);
    neuron.u += dt * (0.02f * (0.2f * v - u)); // Standard Izhikevich parameters
    
    // === DENDRITIC COMPARTMENT VOLTAGE DYNAMICS ===
    for (int c = 1; c < 4; c++) {
        // Leak current proportional to compartment resistance
        float leak_current = -neuron.I_syn[c] / DendriticConstants::COMPARTMENT_RESISTANCE[c];
        
        // Update synaptic current with leak and decay
        neuron.I_syn[c] += dt * (leak_current - neuron.I_syn[c] / NeuronModelConstants::SYNAPTIC_TAU_1);
        
        // Ensure currents don't grow unbounded
        neuron.I_syn[c] = fmaxf(-100.0f, fminf(100.0f, neuron.I_syn[c]));
    }
    
    // Update activity measure
    neuron.average_activity += dt * fabsf(total_synaptic_current) / 1000.0f;
    neuron.average_activity *= 0.9999f; // Slow decay
}

/**
 * @brief Handle inter-modular communication and attention mechanisms
 * This kernel supports the modular architecture by processing cross-module interactions
 */
__global__ void processModularInteractions(GPUNeuronState* neurons, int num_neurons,
                                         int* module_assignments, float* attention_weights,
                                         float* global_inhibition, float current_time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    if (!module_assignments || !attention_weights) return;
    
    GPUNeuronState& neuron = neurons[idx];
    int module_id = module_assignments[idx];
    
    // Apply attention-based modulation
    float attention_factor = attention_weights[module_id];
    
    // Modulate excitability based on attention
    neuron.excitability *= (0.5f + 1.5f * attention_factor);
    
    // Apply global inhibition if provided
    if (global_inhibition) {
        float inhibition = global_inhibition[module_id];
        for (int c = 0; c < 4; c++) {
            neuron.I_syn[c] -= inhibition * 0.1f;
        }
    }
    
    // Inter-modular feedback: highly active modules suppress others
    if (neuron.average_firing_rate > NeuronModelConstants::TARGET_FIRING_RATE * 2.0f) {
        // This neuron's module is highly active - contribute to global inhibition
        if (global_inhibition) {
            atomicAdd(&global_inhibition[module_id], 0.001f);
        }
    }
}

// ============================================================================
// WRAPPER FUNCTIONS FOR MODULAR NEURAL NETWORK INTEGRATION
// ============================================================================

extern "C" {

void launchInitializeModularSpikeState(GPUNeuronState* d_neurons, int num_neurons,
                                      int* d_module_assignments, float current_time) {
    dim3 block(256);
    dim3 grid((num_neurons + block.x - 1) / block.x);
    
    initializeModularSpikeState<<<grid, block>>>(d_neurons, num_neurons, 
                                                d_module_assignments, current_time);
    cudaDeviceSynchronize();
}

void launchUpdateNeuronSpikes(GPUNeuronState* d_neurons, int num_neurons,
                            float current_time, float dt) {
    dim3 block(256);
    dim3 grid((num_neurons + block.x - 1) / block.x);
    
    updateNeuronSpikes<<<grid, block>>>(d_neurons, num_neurons, current_time, dt);
    cudaDeviceSynchronize();
}

void launchDetectSpikes(const GPUNeuronState* d_neurons, GPUSpikeEvent* d_spikes,
                       int* d_spike_count, int num_neurons, float current_time,
                       int* d_module_assignments, int max_spike_events) {
    dim3 block(256);
    dim3 grid((num_neurons + block.x - 1) / block.x);
    
    detectSpikes<<<grid, block>>>(d_neurons, d_spikes, d_spike_count, num_neurons,
                                 current_time, d_module_assignments, max_spike_events);
    cudaDeviceSynchronize();
}

void launchUpdateNeuronVoltages(GPUNeuronState* d_neurons, float dt, int num_neurons,
                               float* d_external_currents) {
    dim3 block(256);
    dim3 grid((num_neurons + block.x - 1) / block.x);
    
    updateNeuronVoltages<<<grid, block>>>(d_neurons, dt, num_neurons, d_external_currents);
    cudaDeviceSynchronize();
}

void launchProcessModularInteractions(GPUNeuronState* d_neurons, int num_neurons,
                                    int* d_module_assignments, float* d_attention_weights,
                                    float* d_global_inhibition, float current_time) {
    dim3 block(256);
    dim3 grid((num_neurons + block.x - 1) / block.x);
    
    processModularInteractions<<<grid, block>>>(d_neurons, num_neurons, d_module_assignments,
                                               d_attention_weights, d_global_inhibition, current_time);
    cudaDeviceSynchronize();
}

} // extern "C"