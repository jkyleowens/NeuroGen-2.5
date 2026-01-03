#pragma once
#include "engine/AgenticSynapse.h"
#include "engine/AgenticNeuron.h"
#include "engine/CorticalColumnAgentic.h"

namespace neurogen {
namespace engine {

// Forward declaration
void launch_synapse_update(
    int num_synapses,
    SynapseParams params,
    SynapseState state,
    NeuronState neuron_state,
    cortical::ColumnState col_state,
    cortical::CorticalColumnAgentic* columns,
    int neurons_per_col,
    cudaStream_t stream = 0
);

} // namespace engine
} // namespace neurogen
