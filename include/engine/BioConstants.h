#pragma once

namespace neurogen {
    namespace bio {
        // ====================================================================
        // BIOLOGICAL PARAMETERS
        // ====================================================================

        // Neuron Type Distribution (based on cortical anatomy)
        constexpr float INHIBITORY_RATIO = 0.20f; // 20% Interneurons (GABA)
        constexpr float EXCITATORY_RATIO = 0.80f; // 80% Pyramidal (Glutamate)

        // ====================================================================
        // CONNECTIVITY PARAMETERS
        // ====================================================================

        // Topographic Connectivity - Gaussian spread for local columns
        constexpr float LOCAL_SPREAD_SIGMA = 2.5f;

        // Lateral Inhibition - Range of Winner-Take-All dynamics
        constexpr float LATERAL_INHIBITION_RADIUS = 4.0f;

        // Connection probability at center of receptive field
        constexpr float MAX_CONNECTION_DENSITY = 0.15f;

        // ====================================================================
        // DALE'S LAW PARAMETERS
        // ====================================================================

        // Weight bounds for Dale's Law enforcement
        constexpr float EXCITATORY_WEIGHT_MIN = 0.001f;
        constexpr float EXCITATORY_WEIGHT_MAX = 2.0f;
        constexpr float INHIBITORY_WEIGHT_MIN = -4.0f;  // Stronger inhibition
        constexpr float INHIBITORY_WEIGHT_MAX = -0.001f;

        // Strength multiplier for inhibitory connections
        constexpr float INHIBITORY_STRENGTH_MULTIPLIER = 2.0f;

        // ====================================================================
        // NEURON TYPE ENUMERATION
        // ====================================================================

        /**
         * @brief Neuron type classification
         *
         * This enum distinguishes between excitatory (pyramidal) and
         * inhibitory (interneuron) cells to enforce Dale's Law:
         * A neuron releases the same neurotransmitter at all synapses.
         */
        enum class NeuronType {
            EXCITATORY = 0, // Pyramidal cells (Glutamate, positive weights only)
            INHIBITORY = 1  // Interneurons (GABA, negative weights only)
        };

        // ====================================================================
        // LAYER CONFIGURATION
        // ====================================================================

        /**
         * @brief Configuration for a cortical layer with spatial dimensions
         */
        struct LayerConfig {
            int width;   // Spatial width (columns)
            int height;  // Spatial height (rows)
            int depth;   // Cortical depth/layers (L2/3, L4, L5/6)

            LayerConfig(int w = 32, int h = 32, int d = 6)
                : width(w), height(h), depth(d) {}
        };

        // ====================================================================
        // NEURON PARAMETERS BY TYPE
        // ====================================================================

        /**
         * @brief Biological parameters for different neuron types
         */
        struct NeuronTypeParams {
            // Threshold parameters
            float threshold_excitatory = 1.0f;   // Standard pyramidal threshold
            float threshold_inhibitory = 0.7f;   // Fast-spiking interneuron (lower)

            // Membrane time constants
            float tau_membrane_excitatory = 20.0f;  // ~20ms (longer integration)
            float tau_membrane_inhibitory = 10.0f;  // ~10ms (faster response)

            // Decay rates (complement of tau)
            float decay_excitatory = 0.95f;  // Long memory
            float decay_inhibitory = 0.80f;  // Short memory (reactive)

            // Refractory periods (ms)
            float refractory_excitatory = 2.0f;
            float refractory_inhibitory = 1.0f;  // Shorter (can fire faster)

            // Adaptation parameters
            float adaptation_excitatory = 0.05f;  // Moderate adaptation
            float adaptation_inhibitory = 0.02f;  // Less adaptation (sustained firing)
        };

        // ====================================================================
        // HELPER FUNCTIONS
        // ====================================================================

        /**
         * @brief Determine neuron type based on index and total count
         *
         * @param neuron_idx Index of the neuron
         * @param num_excitatory Number of excitatory neurons (boundary)
         * @return NeuronType classification
         */
        inline __host__ __device__ NeuronType getNeuronType(int neuron_idx, int num_excitatory) {
            return (neuron_idx < num_excitatory) ? NeuronType::EXCITATORY : NeuronType::INHIBITORY;
        }

        /**
         * @brief Enforce Dale's Law weight constraints
         *
         * @param weight Current weight value
         * @param is_inhibitory Whether the presynaptic neuron is inhibitory
         * @return Clamped weight value
         */
        inline __host__ __device__ float clampWeightDalesLaw(float weight, bool is_inhibitory) {
            if (is_inhibitory) {
                // Must be negative
                if (weight > 0.0f) weight = INHIBITORY_WEIGHT_MAX;
                if (weight < INHIBITORY_WEIGHT_MIN) weight = INHIBITORY_WEIGHT_MIN;
                if (weight > INHIBITORY_WEIGHT_MAX) weight = INHIBITORY_WEIGHT_MAX;
            } else {
                // Must be positive
                if (weight < 0.0f) weight = EXCITATORY_WEIGHT_MIN;
                if (weight < EXCITATORY_WEIGHT_MIN) weight = EXCITATORY_WEIGHT_MIN;
                if (weight > EXCITATORY_WEIGHT_MAX) weight = EXCITATORY_WEIGHT_MAX;
            }
            return weight;
        }

    } // namespace bio
} // namespace neurogen
