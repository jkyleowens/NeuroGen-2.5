#include <engine/EligibilityAndRewardKernels.cuh>
#include <engine/NeuronModelConstants.h>

// --- Kernel to apply reward and consolidate weight changes ---
__global__ void applyRewardKernel(
    GPUSynapse* synapses,
    float reward,
    float dt,
    int num_synapses)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;

    GPUSynapse& synapse = synapses[idx];
    if (synapse.active == 0) return;

    // The change in weight is the eligibility trace modulated by reward and sensitivity.
    float dw = synapse.eligibility_trace * reward * synapse.dopamine_sensitivity * NeuronModelConstants::REWARD_LEARNING_RATE * dt;
    float new_weight = synapse.weight + dw;

    // Clamp the new weight within the synapse's defined min/max bounds.
    synapse.weight = fmaxf(synapse.min_weight, fminf(synapse.max_weight, new_weight));

    // Decay the eligibility trace after it has been "used" by the reward signal.
    synapse.eligibility_trace *= NeuronModelConstants::ELIGIBILITY_TRACE_DECAY;
}


// --- Kernel to adapt the synapse's sensitivity to dopamine ---
__global__ void adaptNeuromodulationKernel(
    GPUSynapse* synapses,
    const GPUNeuronState* neurons,
    float current_dopamine,
    int num_synapses,
    float current_time)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;

    GPUSynapse& synapse = synapses[idx];
    if (synapse.active == 0) return;

    const GPUNeuronState& post_neuron = neurons[synapse.post_neuron_idx];

    // Determine if the postsynaptic neuron spiked in the last time step.
    bool just_spiked = (post_neuron.last_spike_time >= (current_time - 0.1f));
    float activity = just_spiked ? 1.0f : 0.0f;

    // Adapt sensitivity based on how well the neuron's activity predicts reward.
    float expected_reward = synapse.dopamine_sensitivity * activity;
    float activity_error = expected_reward - activity;
    float adaptation_rate = 0.001f;

    synapse.dopamine_sensitivity += adaptation_rate * activity_error * current_dopamine;

    // Clamp sensitivity to a reasonable range [0.1, 2.0].
    synapse.dopamine_sensitivity = fmaxf(0.1f, fminf(2.0f, synapse.dopamine_sensitivity));
}