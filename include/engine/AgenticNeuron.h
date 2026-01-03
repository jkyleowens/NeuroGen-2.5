/**
 * @file AgenticNeuron.h
 * @brief NeuroGen V2 Agentic Neuron Architecture
 * 
 * This file defines the "Agentic Neuron" model, transitioning from OOP to 
 * Data-Oriented Design (SoA) for CUDA optimization.
 * 
 * Key components:
 * - NeuronParams: Read-only parameters (constant memory friendly)
 * - NeuronState: Dynamic state in Structure of Arrays (SoA) layout
 * - AgenticNeuron: Static class implementing ALIF logic with float outputs
 * 
 * @version 2.5-Agentic
 * @date January 2, 2026
 */

#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace neurogen {
namespace engine {

// ============================================================================
// NEURON PARAMETERS (Read-Only)
// ============================================================================
struct NeuronParams {
    // Membrane Dynamics
    float tau_m;            // Membrane time constant (ms)
    float tau_adapt;        // Adaptation time constant (ms)
    float v_rest;           // Resting potential (mV)
    float v_reset;          // Reset potential (mV)
    float v_thresh_base;    // Base firing threshold (mV)
    
    // Adaptation & Refractory
    float adapt_inc;        // Adaptation increment per spike
    float refract_time;     // Refractory period (ms)
    
    // Output Scaling
    float output_scale;     // Scaling factor for float output
    float tau_out;          // Output time constant (ms)
    
    // Pre-computed constants for efficiency
    float decay_m;          // exp(-dt/tau_m)
    float decay_adapt;      // exp(-dt/tau_adapt)
    float decay_out;        // exp(-dt/tau_out)

    __host__ void compute_decays(float dt) {
        decay_m = expf(-dt / tau_m);
        decay_adapt = expf(-dt / tau_adapt);
        decay_out = expf(-dt / tau_out);
    }
};

// ============================================================================
// NEURON STATE (Dynamic) - Structure of Arrays Layout
// ============================================================================
struct NeuronState {
    // Core State
    float* v;               // Membrane potential
    float* a;               // Adaptation variable
    float* v_thresh;        // Dynamic threshold
    
    // Timing & Output
    float* last_spike;      // Time of last spike
    float* output;          // Current float output value
    float* trace;           // Eligibility trace
    
    // Input Integration
    float* i_syn;           // Synaptic input current
    float* i_stim;          // External stimulation current
    
    // Memory Management
    size_t capacity;
    
    __host__ void allocate(size_t n) {
        capacity = n;
        cudaMalloc(&v, n * sizeof(float));
        cudaMalloc(&a, n * sizeof(float));
        cudaMalloc(&v_thresh, n * sizeof(float));
        cudaMalloc(&last_spike, n * sizeof(float));
        cudaMalloc(&output, n * sizeof(float));
        cudaMalloc(&trace, n * sizeof(float));
        cudaMalloc(&i_syn, n * sizeof(float));
        cudaMalloc(&i_stim, n * sizeof(float));
        
        initialize();
    }
    
    __host__ void initialize() {
        cudaMemset(v, 0, capacity * sizeof(float)); // Or v_rest? Usually init to v_rest in kernel, but 0 is safe for now
        cudaMemset(a, 0, capacity * sizeof(float));
        cudaMemset(v_thresh, 0, capacity * sizeof(float)); // Should be v_thresh_base
        cudaMemset(last_spike, 0, capacity * sizeof(float)); // -infinity?
        cudaMemset(output, 0, capacity * sizeof(float));
        cudaMemset(trace, 0, capacity * sizeof(float));
        cudaMemset(i_syn, 0, capacity * sizeof(float));
        cudaMemset(i_stim, 0, capacity * sizeof(float));
    }
    
    __host__ void free() {
        cudaFree(v);
        cudaFree(a);
        cudaFree(v_thresh);
        cudaFree(last_spike);
        cudaFree(output);
        cudaFree(trace);
        cudaFree(i_syn);
        cudaFree(i_stim);
    }
};

// ============================================================================
// AGENTIC NEURON LOGIC
// ============================================================================
class AgenticNeuron {
public:
    // Main update kernel - runs per neuron per tick
    __host__ __device__ static void update_tick(
        int idx,                    // Neuron index
        float t,                    // Current simulation time
        float dt,                   // Time step
        const NeuronParams& params, // Read-only parameters
        NeuronState& state          // Mutable state arrays
    ) {
        // 1. Load State (Coalesced reads if idx is thread index)
        float v = state.v[idx];
        float a = state.a[idx];
        float i_total = state.i_syn[idx] + state.i_stim[idx];
        float last_spike = state.last_spike[idx];
        float trace = state.trace[idx];
        
        // 2. Check Refractory Period
        if (t - last_spike < params.refract_time) {
            // In refractory period: clamp voltage, decay adaptation
            state.v[idx] = params.v_reset;
            state.a[idx] = a * params.decay_adapt;
            state.output[idx] = 0.0f; 
            state.trace[idx] = trace * params.decay_out;
            return;
        }
        
        // 3. Evolve State (Continuous Dynamics)
        // V(t+1) = V(t) * decay_m + V_inf * (1 - decay_m)
        // Note: 'a' is used for adaptive threshold, not subtractive current here.
        float v_inf = params.v_rest + i_total;
        float v_next = v * params.decay_m + v_inf * (1.0f - params.decay_m);
        
        // A(t+1) - Decay
        float a_next = a * params.decay_adapt;
        
        // 4. Threshold & Spiking
        float threshold = params.v_thresh_base + a;
        
        bool spike = v_next >= threshold;
        float output = 0.0f;
        
        if (spike) {
            // Spike Event
            v_next = params.v_reset;
            a_next += params.adapt_inc;
            state.last_spike[idx] = t;
            
            // Float Output (Surrogate Gradient / Soft Spike)
            output = params.output_scale; 
        }
        
        // 5. Eligibility Trace Update
        // e(t+1) = e(t) * decay_out + output
        float trace_next = trace * params.decay_out + output;
        
        // 6. Store State
        state.v[idx] = v_next;
        state.a[idx] = a_next;
        state.v_thresh[idx] = threshold;
        state.output[idx] = output;
        state.trace[idx] = trace_next;
        
        // Reset current for next step
        state.i_syn[idx] = 0.0f;
    }
};

} // namespace engine
} // namespace neurogen
