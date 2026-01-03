#ifndef NEURAL_CONSTANTS_H
#define NEURAL_CONSTANTS_H

// ============================================================================
// NEURAL NETWORK STRUCTURAL CONSTANTS
// ============================================================================

// Compartment model constants
#define MAX_COMPARTMENTS 4          // Maximum number of compartments per neuron (soma + 3 dendrites)

// Receptor and channel constants
#define NUM_RECEPTOR_TYPES 8        // Number of different receptor types

// Dendritic spike constants
#define MAX_DENDRITIC_SPIKES 4      // Maximum dendritic spike history per neuron

// Synapse constants
#define MAX_SYNAPSES_PER_NEURON 1000  // Maximum incoming synapses per neuron

// Network size constants
#define MAX_NEURONS 100000          // Maximum neurons in network
#define MAX_SYNAPSES 10000000       // Maximum total synapses

// Buffer sizes
#define FIRING_RATE_HISTORY_SIZE 100  // Size of firing rate history buffer
#define REWARD_HISTORY_SIZE 50        // Size of reward history buffer

#endif // NEURAL_CONSTANTS_H

