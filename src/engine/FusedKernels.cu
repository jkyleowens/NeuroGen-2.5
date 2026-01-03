#include <engine/FusedKernels.cuh>
#include <engine/NeuronModelConstants.h>

// ============================================================================
// HELPER DEVICE FUNCTIONS
// ============================================================================

__device__ inline float nmda_gating_factor(float V) {
    return 1.0f / (1.0f + expf(-(V + 30.0f) * 0.15f));
}

// ============================================================================
// FUSED NEURON UPDATE KERNEL
// ============================================================================

__global__ void fusedNeuronUpdateKernel(
    NeuronArrays arrays,
    float current_time,
    float dt,
    float dopamine_level,
    float serotonin_level,
    int num_neurons)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;
    
    // === STEP 1: NEURON UPDATE (Izhikevich dynamics) ===
    
    // Check refractory period
    bool refractory = false;
    if (current_time < arrays.last_spike_time[idx] + NeuronModelConstants::ABSOLUTE_REFRACTORY_PERIOD) {
        refractory = true;
    }
    
    float v = arrays.V[idx];
    float u = arrays.u[idx];
    
    if (!refractory) {
        // Gather total synaptic input current (from all compartments)
        float total_current = (arrays.I_syn_0[idx] + arrays.I_syn_1[idx] + 
                              arrays.I_syn_2[idx] + arrays.I_syn_3[idx]) * 
                              arrays.excitability[idx] + arrays.I_ext[idx];
        
        // Izhikevich model dynamics
        // Update membrane potential and recovery variable
        float v_next = v + dt * (0.04f * v * v + 5.0f * v + 140.0f - u + total_current);
        float u_next = u + dt * (0.02f * (0.2f * v - u));
        
        v = v_next;
        u = u_next;
        
        // Check for spike
        if (v >= NeuronModelConstants::SPIKE_THRESHOLD) {
            v = NeuronModelConstants::RESET_POTENTIAL;
            u += 8.0f;
            
            // Update spike times
            arrays.previous_spike_time[idx] = arrays.last_spike_time[idx];
            arrays.last_spike_time[idx] = current_time;
            
            // Update firing rate
            float isi = current_time - arrays.previous_spike_time[idx];
            if (isi > 0.0f) {
                arrays.firing_rate[idx] = 1000.0f / isi; // Convert to Hz
            }
        }
        
        // Store updated state
        arrays.V[idx] = v;
        arrays.u[idx] = u;
    }
    
    // === STEP 2: CALCIUM DIFFUSION ===
    
    float gating = nmda_gating_factor(v);
    
    // Update calcium in each compartment
    float ca0 = arrays.ca_conc_0[idx];
    float ca1 = arrays.ca_conc_1[idx];
    float ca2 = arrays.ca_conc_2[idx];
    float ca3 = arrays.ca_conc_3[idx];
    
    // Voltage-dependent influx from synaptic activity (NMDA)
    float influx0 = arrays.I_syn_0[idx] * gating * 0.2f;
    float influx1 = arrays.I_syn_1[idx] * gating * 0.2f;
    float influx2 = arrays.I_syn_2[idx] * gating * 0.2f;
    float influx3 = arrays.I_syn_3[idx] * gating * 0.2f;
    
    // Back-propagating action potential (BAP)
    bool just_spiked = (arrays.last_spike_time[idx] == current_time);
    if (just_spiked) {
        influx0 += 0.8f;
        influx1 += 0.8f;
        influx2 += 0.8f;
        influx3 += 0.8f;
    }
    
    // Natural decay and update
    arrays.ca_conc_0[idx] = fmaxf(0.0f, ca0 * NeuronModelConstants::CALCIUM_DECAY + influx0 * dt);
    arrays.ca_conc_1[idx] = fmaxf(0.0f, ca1 * NeuronModelConstants::CALCIUM_DECAY + influx1 * dt);
    arrays.ca_conc_2[idx] = fmaxf(0.0f, ca2 * NeuronModelConstants::CALCIUM_DECAY + influx2 * dt);
    arrays.ca_conc_3[idx] = fmaxf(0.0f, ca3 * NeuronModelConstants::CALCIUM_DECAY + influx3 * dt);
    
    // === STEP 3: NEUROMODULATION ===
    
    // Update local neuromodulator concentrations with diffusion
    float target_dopamine = dopamine_level;
    float target_serotonin = serotonin_level;
    
    // Smooth transition to target levels
    arrays.dopamine_concentration[idx] = 0.9f * arrays.dopamine_concentration[idx] + 0.1f * target_dopamine;
    arrays.serotonin_level[idx] = 0.95f * arrays.serotonin_level[idx] + 0.05f * target_serotonin;
    
    // Clamp to physiological ranges
    arrays.dopamine_concentration[idx] = fmaxf(0.0f, fminf(2.0f, arrays.dopamine_concentration[idx]));
    arrays.serotonin_level[idx] = fmaxf(0.0f, fminf(1.5f, arrays.serotonin_level[idx]));
    
    // Modulate neuron excitability based on neuromodulators
    // Dopamine increases excitability
    float dopamine_factor = 1.0f + 0.3f * arrays.dopamine_concentration[idx];
    // Serotonin slightly decreases excitability (inhibitory effect)
    float serotonin_factor = 1.0f - 0.15f * arrays.serotonin_level[idx];
    
    // Fix: Don't multiply excitability in place every time, or it will explode/vanish
    // Instead, assume base excitability is constant or slowly changing, and apply modulation temporarily or to a target
    // For now, just clamp it to be safe, but ideally we should have base_excitability vs current_excitability
    float base_excitability = 1.0f; // Simplified assumption
    arrays.excitability[idx] = base_excitability * dopamine_factor * serotonin_factor;
    arrays.excitability[idx] = fmaxf(0.1f, fminf(3.0f, arrays.excitability[idx]));
}

// ============================================================================
// FUSED PLASTICITY KERNEL
// ============================================================================

__global__ void fusedPlasticityKernel(
    SynapseArrays synapse_arrays,
    NeuronArrays neuron_arrays,
    float reward_signal,
    float current_time,
    float dt,
    int num_synapses)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;
    
    // Check if synapse is active
    if (synapse_arrays.active[idx] == 0) return;
    
    // Get neuron indices
    int pre_idx = synapse_arrays.pre_neuron_idx[idx];
    int post_idx = synapse_arrays.post_neuron_idx[idx];
    
    // Validate indices
    if (pre_idx < 0 || post_idx < 0) return;
    
    // === INPUT PROPAGATION ===
    // Check for pre-synaptic spike
    // Using a small epsilon for floating point comparison
    bool pre_spiked = (fabsf(neuron_arrays.last_spike_time[pre_idx] - current_time) < dt * 1.5f);
    
    if (pre_spiked) {
        // Propagate current to post-synaptic neuron's specific compartment
        int compartment = synapse_arrays.post_compartment[idx];
        float weight = synapse_arrays.effective_weight[idx];
        
        // Simple atomic add to input buffer
        // Note: In a real implementation, we'd need to handle compartment indexing more safely
        // Here we assume 4 compartments per neuron
        // Using atomicAdd because multiple synapses target the same neuron
        if (compartment == 0) atomicAdd(&neuron_arrays.I_syn_0[post_idx], weight);
        else if (compartment == 1) atomicAdd(&neuron_arrays.I_syn_1[post_idx], weight);
        else if (compartment == 2) atomicAdd(&neuron_arrays.I_syn_2[post_idx], weight);
        else if (compartment == 3) atomicAdd(&neuron_arrays.I_syn_3[post_idx], weight);
        
        // Update last pre-spike time for this synapse
        synapse_arrays.last_pre_spike_time[idx] = current_time;
    }
    
    // === STEP 1: STDP WITH SIGN-PRESERVING ELIGIBILITY ===
    
    // Calculate spike timing difference
    float delta_t = neuron_arrays.last_spike_time[post_idx] - neuron_arrays.last_spike_time[pre_idx];
    
    // STDP parameters
    float stdp_window = 20.0f; // 20ms window
    float calcium_factor = neuron_arrays.ca_conc_0[post_idx]; // Use postsynaptic calcium
    
    float stdp_magnitude = 0.0f;
    
    // Check if within STDP window - preserve sign for proper LTP/LTD
    if (delta_t < stdp_window && delta_t > -stdp_window) {
        if (delta_t > 0) {
            // LTP: Post after pre (causal) - positive magnitude
            stdp_magnitude = expf(-delta_t / 10.0f) * calcium_factor;
            synapse_arrays.weight[idx] += stdp_magnitude * synapse_arrays.learning_rate[idx] * dt;
        } else {
            // LTD: Pre before post (anti-causal) - negative magnitude
            stdp_magnitude = -expf(delta_t / 10.0f) * calcium_factor;
            synapse_arrays.weight[idx] += stdp_magnitude * synapse_arrays.learning_rate[idx] * 0.5f * dt;
        }
    }
    
    // === STEP 2: ELIGIBILITY TRACE UPDATE (SIGN-PRESERVING) ===
    
    // Decay existing trace
    float trace_decay = 0.95f;
    float current_trace = synapse_arrays.eligibility_trace[idx] * trace_decay;
    
    // Add new contribution from STDP - PRESERVE SIGN
    current_trace += stdp_magnitude * 0.1f;
    
    // Update trace (allow both positive and negative)
    synapse_arrays.eligibility_trace[idx] = fmaxf(-2.0f, fminf(2.0f, current_trace));
    
    // === STEP 3: REWARD-MODULATED LEARNING ===
    
    // Apply reward modulation if significant
    if (fabsf(reward_signal) > 0.001f) {
        // Three-factor learning rule: reward × trace × learning_rate
        float weight_change = reward_signal * 
                             synapse_arrays.eligibility_trace[idx] * 
                             synapse_arrays.learning_rate[idx] * 
                             dt;
        
        synapse_arrays.weight[idx] += weight_change;
        
        // Update dopamine sensitivity based on reward prediction error
        if (reward_signal > 0.0f) {
            synapse_arrays.dopamine_sensitivity[idx] *= 1.01f; // Increase sensitivity
        } else {
            synapse_arrays.dopamine_sensitivity[idx] *= 0.99f; // Decrease sensitivity
        }
        
        synapse_arrays.dopamine_sensitivity[idx] = fmaxf(0.1f, fminf(2.0f, 
            synapse_arrays.dopamine_sensitivity[idx]));
    }
    
    // === STEP 4: WEIGHT BOUNDS ===
    
    float min_w = synapse_arrays.min_weight[idx];
    float max_w = synapse_arrays.max_weight[idx];
    synapse_arrays.weight[idx] = fmaxf(min_w, fminf(max_w, synapse_arrays.weight[idx]));
    
    // Update effective weight (includes neuromodulator effects)
    float dopamine_modulation = 1.0f + 0.2f * synapse_arrays.dopamine_sensitivity[idx] * 
                                neuron_arrays.dopamine_concentration[post_idx];
    synapse_arrays.effective_weight[idx] = synapse_arrays.weight[idx] * dopamine_modulation;
}
