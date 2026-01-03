#ifndef NEURON_MODEL_CONSTANTS_H
#define NEURON_MODEL_CONSTANTS_H

#include <cuda_runtime.h>

namespace NeuronModelConstants {

    // Basic neuron model constants
    static __device__ constexpr float RESET_POTENTIAL = -65.0f;
    static __device__ constexpr float MEMBRANE_TIME_CONSTANT = 10.0f;
    static __device__ constexpr float MEMBRANE_RESISTANCE = 10.0f;
    static __device__ constexpr float ABSOLUTE_REFRACTORY_PERIOD = 2.0f;

    // Synaptic constants
    static __device__ constexpr float SYNAPTIC_TAU_1 = 0.5f;
    static __device__ constexpr float SYNAPTIC_TAU_2 = 2.0f;

    // STDP constants
    static __device__ constexpr float STDP_A_PLUS = 0.01f;
    static __device__ constexpr float STDP_A_MINUS = 0.012f;
    static __device__ constexpr float STDP_TAU_PLUS = 20.0f;
    static __device__ constexpr float STDP_TAU_MINUS = 20.0f;

    // Weight bounds
    static __device__ constexpr float MAX_WEIGHT = 1.0f;
    static __device__ constexpr float MIN_WEIGHT = 0.01f;

    // Calcium dynamics
    static __device__ constexpr float CALCIUM_DECAY = 0.995f;
    static __device__ constexpr float CA_THRESHOLD_LTP = 1.3f;
    static __device__ constexpr float CA_THRESHOLD_LTD = 1.0f;

    // Homeostatic plasticity
    static __device__ constexpr float TARGET_FIRING_RATE = 5.0f;
    static __device__ constexpr float HOMEOSTATIC_TIMESCALE = 1000.0f; // ms

    // Reinforcement learning
    static __device__ constexpr float ELIGIBILITY_TRACE_DECAY = 0.99f;
    static __device__ constexpr float REWARD_LEARNING_RATE = 0.01f;

    // Neuromodulation effects
    static __device__ constexpr float ACETYLCHOLINE_EXCITABILITY_FACTOR = 0.15f;
    static __device__ constexpr float SEROTONIN_INHIBITORY_FACTOR = 0.2f;
    static __device__ constexpr float ACETYLCHOLINE_PLASTICITY_FACTOR = 0.1f;

    static __device__ constexpr float SPIKE_THRESHOLD = -55.0f;

    static __device__ constexpr float RESTING_POTENTIAL = -70.0f;

} // namespace NeuronModelConstants

#endif // NEURON_MODEL_CONSTANTS_H
