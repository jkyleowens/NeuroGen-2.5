#include <engine/StructuralPlasticityKernels.cuh>
#include <engine/NeuronModelConstants.h>
#include <engine/GPUNeuralStructures.h>
#include <curand_kernel.h>
#include <cmath>

// ============================================================================
// BIOLOGICALLY INSPIRED STRUCTURAL PLASTICITY KERNELS
// ============================================================================

/**
 * @brief Marks weak or underutilized synapses for pruning
 *
 * This kernel implements activity-dependent synaptic pruning, a critical
 * mechanism in biological neural development and learning. Synapses that
 * are both structurally weak and functionally inactive are deactivated,
 * mimicking the "use it or lose it" principle of neural development.
 */
__global__ void markPrunableSynapsesKernel(GPUSynapse* synapses, int num_synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;

    GPUSynapse& synapse = synapses[idx];
    if (synapse.active == 0) return;

    // === BIOLOGICAL PRUNING CRITERIA ===
    
    // Structural weakness: synapse weight below minimum threshold
    bool is_structurally_weak = fabsf(synapse.weight) < (NeuronModelConstants::MIN_WEIGHT * 0.1f);
    
    // Functional inactivity: very low recent activity
    bool is_functionally_inactive = synapse.activity_metric < 0.01f;
    
    // Temporal inactivity: no recent spike activity
    bool is_temporally_inactive = (synapse.last_active_time > 0.0f) && 
                                 (synapse.last_active_time < -1000.0f); // No activity in last second
    
    // === COMPETITIVE ELIMINATION ===
    // Check if this synapse is significantly weaker than its maximum potential
    bool below_competitive_threshold = synapse.weight < (synapse.max_weight * 0.05f);
    
    // Prune if synapse meets multiple weakness criteria
    if ((is_structurally_weak && is_functionally_inactive) || 
        (is_structurally_weak && is_temporally_inactive) ||
        (below_competitive_threshold && is_functionally_inactive)) {
        
        // Deactivate the synapse (biological synaptic elimination)
        synapse.active = 0;
        synapse.weight = 0.0f;
        synapse.effective_weight = 0.0f;
        synapse.eligibility_trace = 0.0f;
    }
}

/**
 * @brief Implements activity-dependent neurogenesis through excitability modulation
 *
 * This kernel simulates neurogenesis by modulating neuronal excitability rather than
 * adding new neurons. In biological systems, new neurons integrate into existing
 * circuits by gradually increasing their functional connectivity and excitability.
 * This approach maintains network stability while enabling structural adaptation.
 */
__global__ void adaptiveNeurogenesisKernel(GPUNeuronState* neurons, const GPUSynapse* synapses,
                                          curandState* rng_states, int num_neurons, int num_synapses) {
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= num_neurons) return;

    GPUNeuronState& neuron = neurons[neuron_idx];
    curandState& local_rng = rng_states[neuron_idx];
    
    // === ASSESS NETWORK COMPUTATIONAL DEMAND ===
    
    // Calculate local network activity in the neuron's neighborhood
    float local_activity = 0.0f;
    float local_weight_saturation = 0.0f;
    int local_connections = 0;
    
    // Sample nearby synapses to assess local network state
    for (int i = 0; i < 50; i++) { // Sample 50 synapses for performance
        int synapse_idx = (int)(curand_uniform(&local_rng) * num_synapses);
        if (synapse_idx < num_synapses && synapses[synapse_idx].active) {
            // Check if this synapse connects to our neuron region
            int connection_distance = abs(synapses[synapse_idx].post_neuron_idx - neuron_idx);
            if (connection_distance < 100) { // Local neighborhood
                local_activity += synapses[synapse_idx].activity_metric;
                local_weight_saturation += synapses[synapse_idx].weight / synapses[synapse_idx].max_weight;
                local_connections++;
            }
        }
    }
    
    if (local_connections > 0) {
        local_activity /= local_connections;
        local_weight_saturation /= local_connections;
    }
    
    // === ADAPTIVE EXCITABILITY MODULATION ===
    
    // Determine if this neuron should increase or decrease its participation
    float current_contribution = neuron.average_activity / NeuronModelConstants::RESTING_POTENTIAL;
    float target_firing_rate = NeuronModelConstants::TARGET_FIRING_RATE;
    
    // High local activity + weight saturation = need for new computational capacity
    bool network_needs_capacity = (local_activity > 0.5f) && (local_weight_saturation > 0.8f);
    
    // Low local activity = potential for network optimization through reduced participation
    bool network_has_excess = (local_activity < 0.1f) && (neuron.average_firing_rate < target_firing_rate * 0.1f);
    
    // === NEUROGENESIS-LIKE ADAPTATION ===
    
    if (network_needs_capacity && neuron.excitability < 2.0f) {
        // "Birth" of functional capacity: gradually increase excitability
        float growth_rate = 0.001f * (1.0f + local_weight_saturation);
        neuron.excitability += growth_rate;
        
        // Reset neuron to optimal starting state for integration
        if (neuron.excitability > 1.5f && neuron.average_firing_rate < 0.1f) {
            neuron.V = NeuronModelConstants::RESTING_POTENTIAL;
            neuron.u = 0.2f * neuron.V;
            neuron.last_spike_time = -1e6f;
            
            // Initialize with small random calcium to promote synapse formation
            for (int c = 0; c < 4; c++) {
                neuron.ca_conc[c] = curand_uniform(&local_rng) * 0.1f;
            }
        }
        
        // Enhance synaptic scaling to facilitate integration
        neuron.synaptic_scaling_factor = fminf(1.5f, neuron.synaptic_scaling_factor * 1.001f);
    }
    
    // === PRUNING-LIKE ADAPTATION ===
    
    else if (network_has_excess && neuron.excitability > 0.1f) {
        // "Death" of functional capacity: gradually decrease excitability
        float decay_rate = 0.0005f;
        neuron.excitability *= (1.0f - decay_rate);
        
        // Reduce synaptic scaling to minimize network interference
        neuron.synaptic_scaling_factor = fmaxf(0.1f, neuron.synaptic_scaling_factor * 0.999f);
        
        // Clear calcium if neuron becomes very inactive
        if (neuron.excitability < 0.2f) {
            for (int c = 0; c < 4; c++) {
                neuron.ca_conc[c] *= 0.95f; // Gradual calcium decay
            }
        }
    }
    
    // === HOMEOSTATIC BOUNDS ===
    
    // Ensure excitability remains within biologically plausible ranges
    neuron.excitability = fmaxf(0.01f, fminf(3.0f, neuron.excitability));
    neuron.synaptic_scaling_factor = fmaxf(0.1f, fminf(2.0f, neuron.synaptic_scaling_factor));
}

/**
 * @brief Promotes formation of new synaptic connections in active network regions
 *
 * This kernel implements activity-dependent synaptogenesis by identifying
 * inactive synapses and reactivating them in regions with high neural activity
 * and learning demand. This mimics the biological process where new synaptic
 * connections form in response to experience and learning.
 */
__global__ void activityDependentSynaptogenesisKernel(GPUSynapse* synapses, const GPUNeuronState* neurons,
                                                     curandState* rng_states, int num_synapses, int num_neurons) {
    int synapse_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (synapse_idx >= num_synapses) return;

    GPUSynapse& synapse = synapses[synapse_idx];
    
    // Only work with inactive synapses (potential sites for new connections)
    if (synapse.active != 0) return;
    
    // === ASSESS LOCAL LEARNING DEMAND ===
    
    // Check activity and plasticity in connected neurons
    if (synapse.pre_neuron_idx >= num_neurons || synapse.post_neuron_idx >= num_neurons) return;
    
    const GPUNeuronState& pre_neuron = neurons[synapse.pre_neuron_idx];
    const GPUNeuronState& post_neuron = neurons[synapse.post_neuron_idx];
    
    // High activity in both pre and post neurons suggests need for connection
    bool high_pre_activity = pre_neuron.average_firing_rate > NeuronModelConstants::TARGET_FIRING_RATE;
    bool high_post_activity = post_neuron.average_firing_rate > NeuronModelConstants::TARGET_FIRING_RATE;
    
    // High calcium in post neuron suggests active plasticity
    float avg_calcium = 0.0f;
    for (int c = 0; c < 4; c++) {
        avg_calcium += post_neuron.ca_conc[c];
    }
    avg_calcium /= 4.0f;
    bool high_plasticity = avg_calcium > 0.5f;
    
    // High excitability suggests neuron is in growth/adaptation mode
    bool growth_mode = (pre_neuron.excitability > 1.2f) || (post_neuron.excitability > 1.2f);
    
    // === SYNAPTOGENESIS CRITERIA ===
    
    if ((high_pre_activity && high_post_activity && high_plasticity) ||
        (growth_mode && (high_pre_activity || high_post_activity))) {
        
        // Initialize new synaptic connection with appropriate random number generator
        curandState& local_rng = rng_states[synapse_idx % num_neurons];
        
        // Reactivate the synapse
        synapse.active = 1;
        
        // Initialize with small but functional weight
        float initial_weight = curand_uniform(&local_rng) * NeuronModelConstants::MIN_WEIGHT * 10.0f;
        synapse.weight = initial_weight;
        synapse.effective_weight = initial_weight;
        
        // Set reasonable weight bounds
        synapse.min_weight = NeuronModelConstants::MIN_WEIGHT;
        synapse.max_weight = NeuronModelConstants::MAX_WEIGHT;
        
        // Initialize plasticity parameters
        synapse.eligibility_trace = 0.0f;
        synapse.plasticity_modulation = 1.0f;
        synapse.dopamine_sensitivity = 0.1f + curand_uniform(&local_rng) * 0.1f;
        synapse.acetylcholine_sensitivity = 0.05f + curand_uniform(&local_rng) * 0.05f;
        
        // Set delay based on distance (simplified)
        int connection_distance = abs(synapse.post_neuron_idx - synapse.pre_neuron_idx);
        synapse.delay = 1.0f + (connection_distance / 100.0f) * 2.0f; // 1-3ms delay
        
        // Initialize activity tracking
        synapse.activity_metric = 0.1f; // Small initial activity to prevent immediate pruning
        synapse.last_active_time = 0.0f;
        synapse.last_pre_spike_time = -1e6f;
        synapse.last_post_spike_time = -1e6f;
        
        // Assign to random compartment (simple distribution)
        synapse.post_compartment = (int)(curand_uniform(&local_rng) * 4.0f) % 4;
    }
}

/**
 * @brief Comprehensive structural plasticity orchestration kernel
 *
 * This kernel coordinates multiple forms of structural plasticity in a single
 * execution, ensuring that pruning, neurogenesis, and synaptogenesis work
 * together harmoniously to maintain network stability while enabling adaptation.
 */
__global__ void coordinatedStructuralPlasticityKernel(GPUNeuronState* neurons, GPUSynapse* synapses,
                                                     curandState* rng_states, int num_neurons, int num_synapses,
                                                     float current_time, float structural_plasticity_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // === GLOBAL NETWORK ASSESSMENT ===
    
    if (idx == 0) {
        // Calculate global network metrics (single thread)
        float total_activity = 0.0f;
        float total_connectivity = 0.0f;
        int active_neurons = 0;
        int active_synapses = 0;
        
        // Sample network state
        for (int i = 0; i < min(1000, num_neurons); i++) {
            if (neurons[i].excitability > 0.1f) {
                total_activity += neurons[i].average_firing_rate;
                active_neurons++;
            }
        }
        
        for (int i = 0; i < min(5000, num_synapses); i++) {
            if (synapses[i].active) {
                total_connectivity += synapses[i].weight;
                active_synapses++;
            }
        }
        
        // Store global metrics in first neuron's unused fields for communication
        if (active_neurons > 0 && active_synapses > 0) {
            float avg_activity = total_activity / active_neurons;
            float avg_connectivity = total_connectivity / active_synapses;
            
            // Use calcium concentration of first neuron to store global state
            neurons[0].ca_conc[3] = avg_activity; // Global activity indicator
        }
    }
    
    __syncthreads();
    
    // === ADAPTIVE STRUCTURAL CHANGES ===
    
    // Access global activity state
    float global_activity = neurons[0].ca_conc[3];
    bool network_overstimulated = global_activity > NeuronModelConstants::TARGET_FIRING_RATE * 2.0f;
    bool network_understimulated = global_activity < NeuronModelConstants::TARGET_FIRING_RATE * 0.5f;
    
    // Apply structural plasticity with global coordination
    if (idx < num_neurons) {
        GPUNeuronState& neuron = neurons[idx];
        
        // Adjust structural plasticity rate based on global network state
        float adapted_rate = structural_plasticity_rate;
        if (network_overstimulated) {
            adapted_rate *= 0.5f; // Reduce plasticity in overstimulated networks
        } else if (network_understimulated) {
            adapted_rate *= 2.0f; // Increase plasticity in understimulated networks
        }
        
        // Apply gradual structural adaptation
        if (neuron.excitability > 0.1f) {
            // Neurons participate in structural adaptation based on their current state
            float activity_factor = neuron.average_firing_rate / NeuronModelConstants::TARGET_FIRING_RATE;
            
            if (activity_factor > 2.0f) {
                // Overactive neuron: reduce participation
                neuron.excitability *= (1.0f - adapted_rate * 0.1f);
                neuron.synaptic_scaling_factor *= (1.0f - adapted_rate * 0.05f);
            } else if (activity_factor < 0.1f) {
                // Underactive neuron: increase participation opportunity
                neuron.excitability *= (1.0f + adapted_rate * 0.1f);
                neuron.synaptic_scaling_factor *= (1.0f + adapted_rate * 0.05f);
            }
        }
        
        // Maintain homeostatic bounds
        neuron.excitability = fmaxf(0.01f, fminf(3.0f, neuron.excitability));
        neuron.synaptic_scaling_factor = fmaxf(0.1f, fminf(2.0f, neuron.synaptic_scaling_factor));
    }
}

// ============================================================================
// HOST WRAPPER FUNCTIONS
// ============================================================================

/**
 * @brief Launch synaptic pruning with optimal CUDA execution parameters
 */
void launchSynapticPruning(GPUSynapse* d_synapses, int num_synapses) {
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    markPrunableSynapsesKernel<<<grid, block>>>(d_synapses, num_synapses);
    cudaDeviceSynchronize();
}

/**
 * @brief Launch neurogenesis simulation with proper resource management
 */
void launchNeurogenesis(GPUNeuronState* d_neurons, const GPUSynapse* d_synapses,
                       curandState* d_rng_states, int num_neurons, int num_synapses) {
    dim3 block(256);
    dim3 grid((num_neurons + block.x - 1) / block.x);
    
    adaptiveNeurogenesisKernel<<<grid, block>>>(d_neurons, d_synapses, d_rng_states, 
                                               num_neurons, num_synapses);
    cudaDeviceSynchronize();
}

/**
 * @brief Launch synaptogenesis with coordinated network-wide effects
 */
void launchSynaptogenesis(GPUSynapse* d_synapses, const GPUNeuronState* d_neurons,
                         curandState* d_rng_states, int num_synapses, int num_neurons) {
    dim3 block(256);
    dim3 grid((num_synapses + block.x - 1) / block.x);
    
    activityDependentSynaptogenesisKernel<<<grid, block>>>(d_synapses, d_neurons, d_rng_states,
                                                          num_synapses, num_neurons);
    cudaDeviceSynchronize();
}

/**
 * @brief Launch coordinated structural plasticity with full system integration
 */
void launchCoordinatedStructuralPlasticity(GPUNeuronState* d_neurons, GPUSynapse* d_synapses,
                                          curandState* d_rng_states, int num_neurons, int num_synapses,
                                          float current_time, float plasticity_rate) {
    dim3 block(256);
    dim3 grid((max(num_neurons, num_synapses) + block.x - 1) / block.x);
    
    coordinatedStructuralPlasticityKernel<<<grid, block>>>(d_neurons, d_synapses, d_rng_states,
                                                          num_neurons, num_synapses, 
                                                          current_time, plasticity_rate);
    cudaDeviceSynchronize();
}