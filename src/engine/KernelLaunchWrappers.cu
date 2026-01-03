#include <engine/KernelLaunchWrappers.cuh>

// Include all the necessary kernel headers
#include <engine/IonChannelInitialization.cuh>
#include <engine/NeuronUpdateKernel.cuh>
#include <engine/CalciumDiffusionKernel.cuh>
#include <engine/EnhancedSTDPKernel.cuh>
#include <engine/EligibilityAndRewardKernels.cuh>
#include <engine/HomeostaticMechanismsKernel.cuh>
#include <engine/NeuronOutputKernel.cuh>
#include <engine/FusedKernels.cuh>

#include <iostream>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

// Helper macro for checking CUDA errors
#define CUDA_CHECK_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA ERROR at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUDA_CHECK_LAST_ERROR() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA KERNEL ERROR at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

namespace KernelLaunchWrappers {

// (Other wrapper implementations remain the same)
void initialize_ion_channels(GPUNeuronState* neurons, int num_neurons) {
    const int num_blocks = (num_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    ionChannelInitializationKernel<<<num_blocks, THREADS_PER_BLOCK>>>(neurons, num_neurons);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

void update_neuron_states(GPUNeuronState* neurons, float current_time, float dt, int num_neurons) {
    // Validate parameters before kernel launch
    if (neurons == nullptr) {
        std::cerr << "[CUDA ERROR] neurons pointer is NULL!" << std::endl;
        return;
    }
    if (num_neurons <= 0) {
        std::cerr << "[CUDA ERROR] invalid num_neurons: " << num_neurons << std::endl;
        return;
    }
    
    const int num_blocks = (num_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    neuronUpdateKernel<<<num_blocks, THREADS_PER_BLOCK>>>(neurons, current_time, dt, num_neurons);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

void update_calcium_dynamics(GPUNeuronState* neurons, float current_time, float dt, int num_neurons) {
    const int num_blocks = (num_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    calciumDiffusionKernel<<<num_blocks, THREADS_PER_BLOCK>>>(neurons, current_time, dt, num_neurons);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

void run_stdp_and_eligibility(
    GPUSynapse* synapses,
    const GPUNeuronState* neurons,
    float current_time,
    float dt,
    int num_synapses)
{
    const int num_blocks = (num_synapses + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    enhancedSTDPKernel<<<num_blocks, THREADS_PER_BLOCK>>>(synapses, neurons, current_time, dt, num_synapses);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

void apply_reward_and_adaptation(
    GPUSynapse* synapses,
    GPUNeuronState* neurons,
    float reward,
    float current_time,
    float dt,
    int num_synapses)
{
    const int num_blocks = (num_synapses + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    applyRewardKernel<<<num_blocks, THREADS_PER_BLOCK>>>(synapses, reward, dt, num_synapses);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    
    adaptNeuromodulationKernel<<<num_blocks, THREADS_PER_BLOCK>>>(synapses, neurons, reward, num_synapses, current_time);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}


// --- FIX: Wrapper function now accepts and passes current_time. ---
void run_homeostatic_mechanisms(
    GPUNeuronState* neurons,
    GPUSynapse* synapses,
    float current_time,
    int num_neurons,
    int num_synapses)
{
    const int neuron_blocks = (num_neurons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // The kernel call now has the correct number of arguments.
    synapticScalingKernel<<<neuron_blocks, THREADS_PER_BLOCK>>>(neurons, synapses, num_neurons, num_synapses, current_time);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    
    intrinsicPlasticityKernel<<<neuron_blocks, THREADS_PER_BLOCK>>>(neurons, num_neurons);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

void compute_neuron_outputs(
    const GPUNeuronState* neurons,
    float* output_buffer,
    int* output_counts,
    int num_neurons,
    int num_outputs,
    int group_size)
{
    if (!neurons || !output_buffer || !output_counts) {
        std::cerr << "[CUDA ERROR] Invalid pointers in compute_neuron_outputs" << std::endl;
        return;
    }
    if (num_neurons <= 0 || num_outputs <= 0) {
        std::cerr << "[CUDA ERROR] Invalid sizes in compute_neuron_outputs" << std::endl;
        return;
    }

    CUDA_CHECK_ERROR(cudaMemset(output_buffer, 0, num_outputs * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemset(output_counts, 0, num_outputs * sizeof(int)));

    const int threads = THREADS_PER_BLOCK;
    int neuron_blocks = (num_neurons + threads - 1) / threads;
    accumulateNeuronOutputsKernel<<<neuron_blocks, threads>>>(
        neurons,
        output_buffer,
        output_counts,
        num_neurons,
        group_size,
        num_outputs);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    int output_blocks = (num_outputs + threads - 1) / threads;
    finalizeNeuronOutputsKernel<<<output_blocks, threads>>>(
        output_buffer,
        output_counts,
        num_outputs);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

// ============================================================================
// FUSED KERNEL WRAPPERS (SoA OPTIMIZED)
// ============================================================================

void launch_fused_neuron_update(
    NeuronArrays* d_neuron_arrays,
    float current_time,
    float dt,
    float dopamine_level,
    float serotonin_level,
    int num_neurons)
{
    if (!d_neuron_arrays || num_neurons <= 0) {
        std::cerr << "[CUDA ERROR] Invalid parameters in launch_fused_neuron_update" << std::endl;
        return;
    }
    
    // Copy structure from device to get array pointers
    NeuronArrays h_arrays;
    CUDA_CHECK_ERROR(cudaMemcpy(&h_arrays, d_neuron_arrays, sizeof(NeuronArrays), cudaMemcpyDeviceToHost));
    
    const int threads = THREADS_PER_BLOCK;
    int blocks = (num_neurons + threads - 1) / threads;
    
    fusedNeuronUpdateKernel<<<blocks, threads>>>(
        h_arrays,
        current_time,
        dt,
        dopamine_level,
        serotonin_level,
        num_neurons
    );
    
    CUDA_CHECK_LAST_ERROR();
    // Note: We don't synchronize here for async execution
}

void launch_fused_plasticity(
    SynapseArrays* d_synapse_arrays,
    NeuronArrays* d_neuron_arrays,
    float reward_signal,
    float current_time,
    float dt,
    int num_synapses)
{
    if (!d_synapse_arrays || !d_neuron_arrays || num_synapses <= 0) {
        std::cerr << "[CUDA ERROR] Invalid parameters in launch_fused_plasticity" << std::endl;
        return;
    }
    
    // Copy structures from device to get array pointers
    SynapseArrays h_syn_arrays;
    NeuronArrays h_neu_arrays;
    CUDA_CHECK_ERROR(cudaMemcpy(&h_syn_arrays, d_synapse_arrays, sizeof(SynapseArrays), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(&h_neu_arrays, d_neuron_arrays, sizeof(NeuronArrays), cudaMemcpyDeviceToHost));
    
    const int threads = THREADS_PER_BLOCK;
    int blocks = (num_synapses + threads - 1) / threads;
    
    fusedPlasticityKernel<<<blocks, threads>>>(
        h_syn_arrays,
        h_neu_arrays,
        reward_signal,
        current_time,
        dt,
        num_synapses
    );
    
    CUDA_CHECK_LAST_ERROR();
    // Note: We don't synchronize here for async execution
}

} // namespace KernelLaunchWrappers