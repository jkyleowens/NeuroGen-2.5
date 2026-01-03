#include "modules/BrainOrchestrator.h"
#include "persistence/CheckpointReader.h"
#include "persistence/CheckpointWriter.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <unordered_map>

BrainOrchestrator::BrainOrchestrator(const Config& config)
    : config_(config),
      should_generate_output_(false),
      current_phase_(CognitivePhase::SENSATION),
      phase_timer_(0.0f),
      total_time_(0.0f),
      time_since_consolidation_(0.0f),
      cognitive_cycles_(0),
      tokens_processed_(0),
      average_reward_(0.0f),
      processing_mode_(config.processing_mode) {
    
    std::cout << "🧠 Initializing Brain Orchestrator..." << std::endl;
    std::cout << "   Processing Mode: " 
              << (processing_mode_ == ProcessingMode::PIPELINED ? "PIPELINED" : "SEQUENTIAL")
              << std::endl;
}

BrainOrchestrator::~BrainOrchestrator() {
    std::cout << "🧹 Brain Orchestrator cleanup complete" << std::endl;
}

void BrainOrchestrator::initializeModules() {
    std::cout << "📦 Initializing cortical modules..." << std::endl;
    
    // 1. Sensory Thalamus - The Gatekeeper (MAXIMALLY SCALED for 4GB GPU)
    {
        CorticalModule::Config config;
        config.module_name = "Thalamus";
        config.num_neurons = 30720;  // 7.5x increase - expanded input pathway
        config.enable_plasticity = true;
        config.learning_rate = 0.0001f;  // REDUCED: Prevent weight explosion
        config.fanout_per_neuron = 64;
        config.num_outputs = 2048;  // 2x increase for richer representations (matches embedding_dim)
        config.num_inputs = 2048;
        config.modulation.dopamine_sensitivity = 0.3f;
        config.modulation.serotonin_sensitivity = 0.5f;
        config.modulation.inhibition_level = 0.1f;  // REDUCED: Allow more neural activity (was 0.2f)
        config.modulation.attention_threshold = 0.2f;  // REDUCED: Let more signals through (was 0.5f)
        config.modulation.excitability_bias = 1.3f;  // INCREASED: Boost responsiveness (was 1.0f)
        
        thalamus_ = std::make_unique<CorticalModule>(config, config_.gpu_device_id);
        module_map_["Thalamus"] = thalamus_.get();
    }
    
    // 2. Wernicke's Area - Language Comprehension (MAXIMALLY SCALED - primary semantic processor)
    {
        CorticalModule::Config config;
        config.module_name = "Wernicke";
        config.num_neurons = 307200;  // 7.5x increase - massive semantic capacity
        config.enable_plasticity = true;
        config.learning_rate = 0.001f;  // REDUCED: Prevent weight saturation (was 0.05f)
        config.fanout_per_neuron = 96;  // Increased connectivity
        config.num_outputs = 10240;  // Scaled output for richer representations
        config.num_inputs = 10240;
        config.modulation.dopamine_sensitivity = 0.4f;
        config.modulation.serotonin_sensitivity = 0.3f;
        config.modulation.inhibition_level = 0.05f;  // REDUCED: Allow more activity (was 0.1f)
        config.modulation.attention_threshold = 0.15f;  // REDUCED: Lower filtering (was 0.2f)
        config.modulation.excitability_bias = 1.4f;  // INCREASED: More responsive (was 1.2f)
        
        wernicke_ = std::make_unique<CorticalModule>(config, config_.gpu_device_id);
        module_map_["Wernicke"] = wernicke_.get();
    }
    
    // 3. Broca's Area - Language Production (MAXIMALLY SCALED - output generation)
    {
        CorticalModule::Config config;
        config.module_name = "Broca";
        config.num_neurons = 307200;  // 7.5x increase - match Wernicke for balanced processing
        config.enable_plasticity = true;
        config.learning_rate = 0.001f;  // REDUCED: Prevent weight saturation (was 0.03f)
        config.fanout_per_neuron = 96;  // Increased connectivity
        config.num_outputs = 8192;  // Optimized for GTX 1650 (4GB VRAM)
        config.num_inputs = 8192;   // Optimized for GTX 1650 (4GB VRAM)
        config.modulation.dopamine_sensitivity = 0.5f;
        config.modulation.serotonin_sensitivity = 0.4f;
        config.modulation.inhibition_level = 0.1f;  // REDUCED: Allow more activity (was 0.2f, originally 0.8f)
        config.modulation.attention_threshold = 0.2f;  // REDUCED: Lower threshold (was 0.3f)
        config.modulation.excitability_bias = 1.0f;  // INCREASED: Boost output (was 0.8f)
        
        broca_ = std::make_unique<CorticalModule>(config, config_.gpu_device_id);
        module_map_["Broca"] = broca_.get();
    }
    
    // 4. Hippocampal Formation - Episodic Memory (MAXIMALLY SCALED - memory capacity)
    {
        CorticalModule::Config config;
        config.module_name = "Hippocampus";
        config.num_neurons = 153600;  // 7.5x increase - much longer context memory
        config.enable_plasticity = true;
        config.learning_rate = 0.002f;  // REDUCED: More conservative (was 0.005f)
        config.fanout_per_neuron = 96;  // Increased connectivity for associations
        config.num_outputs = 8192;  // Scaled memory representations
        config.num_inputs = 8192;
        config.modulation.dopamine_sensitivity = 0.6f;
        config.modulation.serotonin_sensitivity = 0.2f;
        config.modulation.inhibition_level = 0.1f;   // REDUCED: Allow more activity (was 0.15f)
        config.modulation.attention_threshold = 0.1f; // REDUCED: Lower filtering (was 0.15f)
        config.modulation.excitability_bias = 1.4f;  // INCREASED: More responsive (was 1.3f)
        
        hippocampus_ = std::make_unique<CorticalModule>(config, config_.gpu_device_id);
        module_map_["Hippocampus"] = hippocampus_.get();
    }
    
    // 5. Prefrontal Cortex - Executive Control (MAXIMALLY SCALED - working memory & integration)
    {
        CorticalModule::Config config;
        config.module_name = "PFC";
        config.num_neurons = 307200;  // 7.5x increase - massive reasoning capacity
        config.enable_plasticity = true;
        config.learning_rate = 0.001f;  // REDUCED: Prevent weight saturation (was 0.01f)
        config.fanout_per_neuron = 128;  // Very high connectivity for integration
        config.num_outputs = 10240;  // Scaled for complex representations
        config.num_inputs = 10240;
        config.modulation.dopamine_sensitivity = 0.5f;
        config.modulation.serotonin_sensitivity = 0.6f;
        config.modulation.inhibition_level = 0.1f;  // REDUCED: Allow more activity (was 0.2f)
        config.modulation.attention_threshold = 0.15f; // REDUCED: Lower filtering (was 0.25f)
        config.modulation.excitability_bias = 1.2f;   // INCREASED: More responsive (was 1.0f)
        
        pfc_ = std::make_unique<CorticalModule>(config, config_.gpu_device_id);
        module_map_["PFC"] = pfc_.get();
    }
    
    // 6. Basal Ganglia - Action Selection (MAXIMALLY SCALED - decision making)
    {
        CorticalModule::Config config;
        config.module_name = "BasalGanglia";
        config.num_neurons = 76800;  // 7.5x increase - sophisticated action selection
        config.enable_plasticity = true;
        config.learning_rate = 0.001f;  // REDUCED: Conservative learning (was 0.002f)
        config.fanout_per_neuron = 96;  // Increased connectivity
        config.num_outputs = 4096;  // Scaled decision space
        config.num_inputs = 4096;
        config.modulation.dopamine_sensitivity = 1.0f;  // Very sensitive to dopamine!
        config.modulation.serotonin_sensitivity = 0.3f;
        config.modulation.inhibition_level = 0.15f;  // REDUCED: Allow more activity (was 0.3f)
        config.modulation.attention_threshold = 0.15f; // REDUCED: Lower filtering (was 0.2f)
        config.modulation.excitability_bias = 1.2f;   // INCREASED: More responsive (was 1.1f)
        
        basal_ganglia_ = std::make_unique<CorticalModule>(config, config_.gpu_device_id);
        module_map_["BasalGanglia"] = basal_ganglia_.get();
    }
    
    std::cout << "✓ All modules initialized successfully" << std::endl;
}

void BrainOrchestrator::createConnectome() {
    std::cout << "🔗 Creating inter-module connections..." << std::endl;
    
    // Connection helper lambda
    auto addConnection = [this](const std::string& name, const std::string& source, 
                               const std::string& target, float strength, 
                               bool excitatory, float threshold, float plasticity) {
        InterModuleConnection::Config config;
        config.connection_name = name;
        config.source_module = module_map_[source];
        config.target_module = module_map_[target];
        config.initial_strength = strength;
        config.is_excitatory = excitatory;
        config.gating_threshold = threshold;
        config.plasticity_rate = plasticity;
        config.enable_plasticity = true;
        
        connections_.push_back(std::make_unique<InterModuleConnection>(config));
    };
    
    // Sensory pathway: Input → Thalamus → Wernicke's
    addConnection("Input_to_Thalamus", "Thalamus", "Wernicke", 1.0f, true, 0.1f, 0.02f);
    
    // Semantic to working memory: Wernicke's → PFC
    addConnection("Wernicke_to_PFC", "Wernicke", "PFC", 0.8f, true, 0.05f, 0.03f);
    
    // Memory encoding: Wernicke's → Hippocampus
    addConnection("Wernicke_to_Hippocampus", "Wernicke", "Hippocampus", 0.9f, true, 0.05f, 0.05f);
    
    // Memory retrieval: Hippocampus → PFC
    addConnection("Hippocampus_to_PFC", "Hippocampus", "PFC", 0.7f, true, 0.1f, 0.04f);
    
    // Executive to action: PFC → Basal Ganglia
    addConnection("PFC_to_BasalGanglia", "PFC", "BasalGanglia", 1.0f, true, 0.05f, 0.03f);
    
    // Action gating: Basal Ganglia → Broca's (Go/No-Go)
    addConnection("BasalGanglia_to_Broca", "BasalGanglia", "Broca", 0.5f, true, 0.2f, 0.02f);
    
    // Semantic to output: PFC → Broca's
    addConnection("PFC_to_Broca", "PFC", "Broca", 0.8f, true, 0.1f, 0.03f);
    
    // Top-down attention: PFC → Thalamus (inhibitory control)
    addConnection("PFC_to_Thalamus", "PFC", "Thalamus", 0.6f, false, 0.1f, 0.02f);
    
    // Top-down modulation: PFC → Wernicke's
    addConnection("PFC_to_Wernicke", "PFC", "Wernicke", 0.5f, true, 0.1f, 0.02f);
    
    // Recurrent PFC (working memory maintenance)
    addConnection("PFC_recurrent", "PFC", "PFC", 0.7f, true, 0.05f, 0.01f);
    
    std::cout << "✓ Created " << connections_.size() << " inter-module connections" << std::endl;
}

std::vector<float> BrainOrchestrator::cognitiveStep(
    const std::vector<float>& input_embedding,
    int target_token_id,
    GPUDecoder* decoder) {
    // Route to appropriate processing mode
    if (processing_mode_ == ProcessingMode::PIPELINED) {
        return pipelinedCognitiveStep(input_embedding, target_token_id, decoder);
    } else if (processing_mode_ == ProcessingMode::BIOLOGICAL_SYNC) {
        return biologicalSynchronousStep(input_embedding, target_token_id, decoder);
    }

    // Sequential mode (original implementation)
    pending_input_ = input_embedding;
    std::vector<float> output;
    
    // Execute the current phase
    switch (current_phase_) {
        case CognitivePhase::SENSATION:
            executeSensationPhase(input_embedding);
            break;
        case CognitivePhase::PERCEPTION:
            executePerceptionPhase();
            break;
        case CognitivePhase::INTEGRATION:
            executeIntegrationPhase();
            break;
        case CognitivePhase::SELECTION:
            executeSelectionPhase();
            break;
        case CognitivePhase::ACTION:
            output = executeActionPhase();
            break;
    }
    
    // Update all modules
    for (auto& [name, module] : module_map_) {
        module->update(config_.time_step_ms, average_reward_);
    }
    
    // Route signals between modules
    routeSignals();
    
    // Update timers
    phase_timer_ += config_.time_step_ms;
    total_time_ += config_.time_step_ms;
    time_since_consolidation_ += config_.time_step_ms;
    
    // Check if we should advance to next phase
    if (phase_timer_ >= getPhaseDuration(current_phase_)) {
        advancePhase();
    }
    
    // Periodic memory consolidation
    if (config_.enable_consolidation && 
        time_since_consolidation_ >= config_.consolidation_interval_ms) {
        consolidateMemory();
        time_since_consolidation_ = 0.0f;
    }
    
    // SAFETY BREAK: Ensure output doesn't block indefinitely
    // If we're in ACTION phase but no output generated yet, 
    // force a break if we've been here too long to prevent infinite loops
    if (current_phase_ == CognitivePhase::ACTION && phase_timer_ > getPhaseDuration(current_phase_) * 2.0f) {
         should_generate_output_ = false;
         advancePhase();
    }
    
    return output;
}

void BrainOrchestrator::executeSensationPhase(const std::vector<float>& input) {
    // Thalamus receives input and evaluates novelty
    thalamus_->receiveInput(input);
    
    // Calculate signal-to-noise ratio for gating
    float snr = thalamus_->calculateSignalToNoise(input);
    
    // Apply gating based on attention threshold
    auto gated_signal = thalamus_->gateSignal(input, 
        thalamus_->getConfig().modulation.attention_threshold);
    
    // Only strong signals pass to cortex
    if (snr > thalamus_->getConfig().modulation.attention_threshold) {
        wernicke_->receiveInput(gated_signal);
    }
}

void BrainOrchestrator::executePerceptionPhase() {
    // Wernicke's Area processes semantic content
    auto semantic_output = wernicke_->getOutputState();
    
    // Send to PFC and Hippocampus
    pfc_->receiveInput(semantic_output);
    hippocampus_->receiveInput(semantic_output);
}

void BrainOrchestrator::executeIntegrationPhase() {
    // PFC integrates semantic info with working memory
    auto pfc_state = pfc_->getOutputState();
    auto working_mem = pfc_->getWorkingMemory();
    
    // Hippocampus attempts memory retrieval based on current pattern
    auto hippocampus_output = hippocampus_->getOutputState();
    
    // Inject retrieved memories into PFC
    pfc_->receiveInput(hippocampus_output);
    
    // Update working memory
    pfc_->setWorkingMemory(pfc_state);
}

void BrainOrchestrator::executeSelectionPhase() {
    // Basal Ganglia evaluates PFC state
    auto pfc_state = pfc_->getOutputState();
    basal_ganglia_->receiveInput(pfc_state);
    
    // Calculate "readiness" to output based on Basal Ganglia activity
    auto bg_output = basal_ganglia_->getOutputState();
    float bg_activity = std::accumulate(bg_output.begin(), bg_output.end(), 0.0f) / 
                       bg_output.size();
    
    // Decision: generate output or wait for more input?
    should_generate_output_ = (bg_activity > 0.5f);
    
    if (should_generate_output_) {
        // "Go" signal - disinhibit Broca's Area
        broca_->applyTopDownBias(0.8f);  // Reduce inhibition
    } else {
        // "No-Go" signal - maintain inhibition
        broca_->applyTopDownBias(0.0f);
    }
}

std::vector<float> BrainOrchestrator::executeActionPhase() {
    std::vector<float> output;
    
    if (should_generate_output_) {
        // Get semantic vector from PFC
        auto pfc_output = pfc_->getOutputState();
        
        // Broca's Area generates output token probabilities
        broca_->receiveInput(pfc_output);
        output = broca_->getOutputState();
        
        // Reset output flag
        should_generate_output_ = false;
        tokens_processed_++;
    }
    
    return output;
}

void BrainOrchestrator::routeSignals() {
    // Transmit signals through all connections
    for (auto& connection : connections_) {
        connection->transmit(config_.time_step_ms);
    }
}

void BrainOrchestrator::distributeReward(float reward) {
    // Update running average
    average_reward_ = 0.95f * average_reward_ + 0.05f * reward;
    
    // Distribute dopamine signal to all modules
    for (auto& [name, module] : module_map_) {
        // Dopamine = reward prediction error
        module->modulate(reward, 0.0f);
    }
    
    // Update connection plasticity based on reward
    updateConnectionPlasticity(reward);
}

std::vector<float> BrainOrchestrator::getGlobalWorkspace() {
    // Combine activity from key modules into global workspace
    std::vector<float> workspace;
    
    auto wernicke_out = wernicke_->getOutputState();
    auto pfc_out = pfc_->getOutputState();
    auto hippocampus_out = hippocampus_->getOutputState();
    
    // Concatenate key module outputs
    workspace.reserve(wernicke_out.size() + pfc_out.size() + hippocampus_out.size());
    workspace.insert(workspace.end(), wernicke_out.begin(), wernicke_out.end());
    workspace.insert(workspace.end(), pfc_out.begin(), pfc_out.end());
    workspace.insert(workspace.end(), hippocampus_out.begin(), hippocampus_out.end());
    
    return workspace;
}

void BrainOrchestrator::updateConnectionPlasticity(float reward) {
    for (auto& connection : connections_) {
        // Get source and target activity levels
        // This is simplified - in reality we'd track actual activities
        float source_activity = 0.5f;  // Placeholder
        float target_activity = 0.5f;  // Placeholder
        
        connection->updatePlasticity(source_activity, target_activity, 
                                    reward, config_.time_step_ms);
    }
}

void BrainOrchestrator::consolidateMemory() {
    // Hippocampal replay: reactivate high-reward sequences
    // This strengthens cortical connections through repeated activation
    
    std::cout << "💾 Running memory consolidation..." << std::endl;
    
    // In a real implementation, this would trigger rapid replay of
    // hippocampal sequences into cortex
}

void BrainOrchestrator::advancePhase() {
    phase_timer_ = 0.0f;
    
    switch (current_phase_) {
        case CognitivePhase::SENSATION:
            current_phase_ = CognitivePhase::PERCEPTION;
            break;
        case CognitivePhase::PERCEPTION:
            current_phase_ = CognitivePhase::INTEGRATION;
            break;
        case CognitivePhase::INTEGRATION:
            current_phase_ = CognitivePhase::SELECTION;
            break;
        case CognitivePhase::SELECTION:
            current_phase_ = CognitivePhase::ACTION;
            break;
        case CognitivePhase::ACTION:
            current_phase_ = CognitivePhase::SENSATION;
            cognitive_cycles_++;
            break;
    }
}

float BrainOrchestrator::getPhaseDuration(CognitivePhase phase) const {
    // Biologically inspired phase durations
    switch (phase) {
        case CognitivePhase::SENSATION:   return 50.0f;   // 0-50ms
        case CognitivePhase::PERCEPTION:  return 100.0f;  // 50-150ms
        case CognitivePhase::INTEGRATION: return 150.0f;  // 150-300ms
        case CognitivePhase::SELECTION:   return 100.0f;  // 300-400ms
        case CognitivePhase::ACTION:      return 100.0f;  // 400ms+
        default: return 100.0f;
    }
}

CorticalModule* BrainOrchestrator::getModule(const std::string& name) {
    auto it = module_map_.find(name);
    return (it != module_map_.end()) ? it->second : nullptr;
}

BrainOrchestrator::Stats BrainOrchestrator::getStats() const {
    Stats stats;
    stats.total_time_ms = total_time_;
    stats.cognitive_cycles = cognitive_cycles_;
    stats.tokens_processed = tokens_processed_;
    stats.average_reward = average_reward_;
    stats.current_phase = current_phase_;
    
    // Get activity levels for each module
    for (const auto& [name, module] : module_map_) {
        auto output = module->getOutputState();
        float activity = output.empty() ? 0.0f : 
            std::accumulate(output.begin(), output.end(), 0.0f) / output.size();
        
        ModuleStats mod_stats;
        mod_stats.activity_level = activity;
        mod_stats.dopamine_level = module->getDopamineLevel();
        mod_stats.serotonin_level = module->getSerotoninLevel();
        
        stats.module_stats[name] = mod_stats;
    }
    
    return stats;
}

persistence::BrainSnapshot BrainOrchestrator::captureSnapshot() const {
    persistence::BrainSnapshot snapshot;
    snapshot.format_version = persistence::kCheckpointFormatVersion;
    snapshot.training_step = static_cast<uint64_t>(
        total_time_ / std::max(0.001f, config_.time_step_ms)
    );
    snapshot.cognitive_cycles = static_cast<uint64_t>(cognitive_cycles_);
    snapshot.tokens_processed = static_cast<uint64_t>(tokens_processed_);
    snapshot.average_reward = average_reward_;
    snapshot.time_since_consolidation_ms = time_since_consolidation_;
    snapshot.modules.reserve(module_map_.size());

    uint32_t module_index = 0;
    for (const auto& [name, module] : module_map_) {
        if (!module) {
            continue;
        }

        persistence::ModuleSnapshot module_snapshot;
        module_snapshot.module_index = module_index++;

        const auto& config = module->getConfig();
        module_snapshot.config.module_name = config.module_name;
        module_snapshot.config.num_neurons = config.num_neurons;
        module_snapshot.config.enable_plasticity = config.enable_plasticity;
        module_snapshot.config.learning_rate = config.learning_rate;
        module_snapshot.config.fanout_per_neuron = config.fanout_per_neuron;
        module_snapshot.config.num_inputs = config.num_inputs;
        module_snapshot.config.num_outputs = config.num_outputs;
        module_snapshot.config.dopamine_sensitivity = config.modulation.dopamine_sensitivity;
        module_snapshot.config.serotonin_sensitivity = config.modulation.serotonin_sensitivity;
        module_snapshot.config.inhibition_level = config.modulation.inhibition_level;
        module_snapshot.config.attention_threshold = config.modulation.attention_threshold;
        module_snapshot.config.excitability_bias = config.modulation.excitability_bias;

        module_snapshot.dopamine_level = module->getDopamineLevel();
        module_snapshot.serotonin_level = module->getSerotoninLevel();
        module_snapshot.working_memory = module->getWorkingMemory();

        // Use CorticalModule wrappers to avoid NetworkCUDA dependency here
        module_snapshot.neurons = module->getNeuronStates();
        module_snapshot.synapses = module->getSynapseStates();

        snapshot.modules.push_back(std::move(module_snapshot));
    }

    snapshot.connections.reserve(connections_.size());
    for (const auto& connection_ptr : connections_) {
        if (!connection_ptr) {
            continue;
        }

        const auto& config = connection_ptr->getConfig();
        persistence::ConnectionSnapshot connection_snapshot;
        connection_snapshot.name = config.connection_name;
        connection_snapshot.source_module = config.source_module ? config.source_module->getName() : "";
        connection_snapshot.target_module = config.target_module ? config.target_module->getName() : "";
        connection_snapshot.is_excitatory = config.is_excitatory;
        connection_snapshot.plasticity_enabled = config.enable_plasticity;
        connection_snapshot.gating_threshold = config.gating_threshold;
        connection_snapshot.plasticity_rate = config.plasticity_rate;
        connection_snapshot.current_strength = connection_ptr->getStrength();
        connection_snapshot.attention_modulation = connection_ptr->getAttentionModulation();

        auto stats = connection_ptr->getStats();
        connection_snapshot.average_activity = stats.average_activity;
        connection_snapshot.total_transmitted = stats.total_transmitted;
        connection_snapshot.activation_count = stats.activation_count;
        connection_snapshot.pre_synaptic_trace = connection_ptr->getPreSynapticTrace();
        connection_snapshot.post_synaptic_trace = connection_ptr->getPostSynapticTrace();

        snapshot.connections.push_back(std::move(connection_snapshot));
    }

    return snapshot;
}

bool BrainOrchestrator::saveCheckpoint(const std::string& file_path) const {
    persistence::CheckpointWriter writer(file_path);
    auto snapshot = captureSnapshot();
    
    // Initialize and write the checkpoint
    if (!writer.initialize(snapshot)) {
        std::cerr << "❌ Failed to initialize checkpoint: " << file_path << std::endl;
        return false;
    }
    
    if (!writer.writeMetadataSection(snapshot)) {
        std::cerr << "❌ Failed to write metadata section: " << file_path << std::endl;
        return false;
    }
    
    // Write neurons
    if (!writer.beginNeuronSection(snapshot.modules.size())) {
        std::cerr << "❌ Failed to begin neuron section: " << file_path << std::endl;
        return false;
    }
    for (const auto& module : snapshot.modules) {
        if (!writer.writeModuleNeurons(module.module_index, module.neurons)) {
            std::cerr << "❌ Failed to write neurons for module: " << module.config.module_name << std::endl;
            return false;
        }
    }
    if (!writer.endNeuronSection()) {
        std::cerr << "❌ Failed to end neuron section: " << file_path << std::endl;
        return false;
    }
    
    // Write synapses
    if (!writer.beginSynapseSection(snapshot.modules.size())) {
        std::cerr << "❌ Failed to begin synapse section: " << file_path << std::endl;
        return false;
    }
    for (const auto& module : snapshot.modules) {
        if (!writer.writeModuleSynapses(module.module_index, module.synapses)) {
            std::cerr << "❌ Failed to write synapses for module: " << module.config.module_name << std::endl;
            return false;
        }
    }
    if (!writer.endSynapseSection()) {
        std::cerr << "❌ Failed to end synapse section: " << file_path << std::endl;
        return false;
    }
    
    // Write optimizer and random state
    if (!writer.writeOptimizerSection(snapshot)) {
        std::cerr << "❌ Failed to write optimizer section: " << file_path << std::endl;
        return false;
    }
    
    if (!writer.writeRandomStateSection(snapshot)) {
        std::cerr << "❌ Failed to write random state section: " << file_path << std::endl;
        return false;
    }
    
    // Finalize
    if (!writer.finalize()) {
        std::cerr << "❌ Failed to finalize checkpoint: " << file_path << std::endl;
        return false;
    }
    
    return true;
}

bool BrainOrchestrator::loadCheckpoint(const std::string& file_path) {
    persistence::CheckpointReader reader(file_path);
    auto snapshot_opt = reader.read();
    if (!snapshot_opt) {
        std::cerr << "❌ Failed to load checkpoint: " << file_path << std::endl;
        return false;
    }

    const auto& snapshot = *snapshot_opt;
    bool success = true;

    for (const auto& module_snapshot : snapshot.modules) {
        auto it = module_map_.find(module_snapshot.config.module_name);
        if (it == module_map_.end()) {
            std::cerr << "⚠️  Missing module for checkpoint data: "
                      << module_snapshot.config.module_name << std::endl;
            success = false;
            continue;
        }

        CorticalModule* module = it->second;
        
        // Use CorticalModule wrappers to avoid NetworkCUDA dependency here
        if (!module_snapshot.neurons.empty() &&
            !module->setNeuronStates(module_snapshot.neurons)) {
            success = false;
        }

        if (!module_snapshot.synapses.empty() &&
            !module->setSynapseStates(module_snapshot.synapses)) {
            success = false;
        }

        module->setWorkingMemory(module_snapshot.working_memory);
        module->restoreNeuromodulatorLevels(module_snapshot.dopamine_level,
                                            module_snapshot.serotonin_level);
    }

    std::unordered_map<std::string, InterModuleConnection*> connection_lookup;
    for (const auto& connection : connections_) {
        if (connection) {
            connection_lookup[connection->getConfig().connection_name] = connection.get();
        }
    }

    for (const auto& connection_snapshot : snapshot.connections) {
        auto it = connection_lookup.find(connection_snapshot.name);
        if (it == connection_lookup.end()) {
            std::cerr << "⚠️  Missing connection for checkpoint data: "
                      << connection_snapshot.name << std::endl;
            success = false;
            continue;
        }

        it->second->restoreState(connection_snapshot.current_strength,
                                 connection_snapshot.attention_modulation,
                                 connection_snapshot.pre_synaptic_trace,
                                 connection_snapshot.post_synaptic_trace,
                                 connection_snapshot.average_activity,
                                 connection_snapshot.total_transmitted,
                                 connection_snapshot.activation_count);
    }

    total_time_ = static_cast<float>(snapshot.training_step) * config_.time_step_ms;
    cognitive_cycles_ = static_cast<int>(snapshot.cognitive_cycles);
    tokens_processed_ = static_cast<int>(snapshot.tokens_processed);
    average_reward_ = snapshot.average_reward;
    time_since_consolidation_ = snapshot.time_since_consolidation_ms;
    phase_timer_ = 0.0f;
    should_generate_output_ = false;
    pending_input_.clear();
    current_phase_ = CognitivePhase::SENSATION;

    if (success) {
        std::cout << "✅ Checkpoint loaded: " << file_path << std::endl;
    }
    return success;
}

void BrainOrchestrator::reset() {
    current_phase_ = CognitivePhase::SENSATION;
    phase_timer_ = 0.0f;
    total_time_ = 0.0f;
    cognitive_cycles_ = 0;
    tokens_processed_ = 0;
    average_reward_ = 0.0f;
    should_generate_output_ = false;
    time_since_consolidation_ = 0.0f;
    
    // Note: Removed verbose logging - reset is called frequently during pre-training
}

std::vector<float> BrainOrchestrator::getBrocaOutput() {
    if (!broca_) {
        return std::vector<float>();
    }
    return broca_->getOutputState();
}

void BrainOrchestrator::modulateGlobalState(float dopamine, float serotonin, float norepinephrine) {
    // Distribute neuromodulators to all modules
    for (auto& pair : module_map_) {
        CorticalModule* module = pair.second;
        if (module) {
            // Apply dopamine and serotonin modulation
            module->modulate(dopamine, serotonin);
            
            // Norepinephrine affects attention/arousal (adjust top-down bias)
            float attention_boost = norepinephrine * 0.5f;
            module->applyTopDownBias(attention_boost);
        }
    }
}

void BrainOrchestrator::setProcessingMode(ProcessingMode mode) {
    processing_mode_ = mode;

    // Reset pipeline state when switching modes
    if (mode == ProcessingMode::PIPELINED) {
        pipeline_state_ = PipelineState();
        std::cout << "✓ Switched to PIPELINED processing mode" << std::endl;
    } else if (mode == ProcessingMode::BIOLOGICAL_SYNC) {
        std::cout << "✓ Switched to BIOLOGICAL_SYNC processing mode (Tick-Tock)" << std::endl;
    } else {
        std::cout << "✓ Switched to SEQUENTIAL processing mode" << std::endl;
    }
}

// ============================================================================
// PIPELINED PROCESSING IMPLEMENTATION
// ============================================================================

std::vector<float> BrainOrchestrator::pipelinedCognitiveStep(
    const std::vector<float>& input_embedding,
    int target_token_id,
    GPUDecoder* decoder) {
    (void)target_token_id;  // Unused in pipelined mode
    (void)decoder;          // Unused in pipelined mode
    std::vector<float> output;
    
    // STAGE 1: Fast Input Encoding (50ms equivalent)
    // Thalamus rapidly encodes new token to working memory
    // Create prediction vector (placeholder for now)
    std::vector<float> prediction(input_embedding.size(), 0.0f);
    fastInputEncoding(input_embedding, prediction);
    
    // STAGE 2: Parallel Cortical Processing
    // Modules operate on PREVIOUS working memory while new input is being encoded
    parallelCorticalProcessing();
    
    // STAGE 3: Conditional Output Generation
    // Broca outputs only when Basal Ganglia signals "ready"
    output = conditionalOutputGeneration();
    
    // STAGE 4: Update Recurrent State
    // Maintain PFC and Hippocampus hidden states for temporal context
    updateRecurrentState();
    
    // Update timers
    total_time_ += config_.time_step_ms;
    time_since_consolidation_ += config_.time_step_ms;
    pipeline_state_.accumulated_processing_time += config_.time_step_ms;
    
    // Periodic memory consolidation
    if (config_.enable_consolidation && 
        time_since_consolidation_ >= config_.consolidation_interval_ms) {
        consolidateMemory();
        time_since_consolidation_ = 0.0f;
    }
    
    return output;
}

void BrainOrchestrator::fastInputEncoding(const std::vector<float>& input, const std::vector<float>& prediction) {
    (void)prediction;  // Reserved for future predictive coding implementation
    
    // Thalamus receives and gates input
    thalamus_->receiveInput(input);
    
    // Calculate signal-to-noise ratio for gating
    float snr = thalamus_->calculateSignalToNoise(input);
    
    // Apply gating based on attention threshold
    auto gated_signal = thalamus_->gateSignal(input, 
        thalamus_->getConfig().modulation.attention_threshold);
    
    // Update thalamus (fast: ~10-20ms biological equivalent)
    thalamus_->update(config_.time_step_ms, average_reward_);
    
    // Shift working memory buffer
    pipeline_state_.wm_previous = pipeline_state_.wm_current;
    
    // Encode to current working memory
    if (snr > thalamus_->getConfig().modulation.attention_threshold) {
        pipeline_state_.wm_current = thalamus_->getOutputState();
    } else {
        // Low SNR: inject minimal signal
        pipeline_state_.wm_current = gated_signal;
    }
    
    // Accumulate context (sliding window)
    if (pipeline_state_.wm_context.empty()) {
        pipeline_state_.wm_context = pipeline_state_.wm_current;
    } else {
        // Exponential moving average of context
        float decay = 0.9f;
        for (size_t i = 0; i < std::min(pipeline_state_.wm_context.size(), 
                                        pipeline_state_.wm_current.size()); ++i) {
            pipeline_state_.wm_context[i] = 
                decay * pipeline_state_.wm_context[i] + 
                (1.0f - decay) * pipeline_state_.wm_current[i];
        }
    }
}

void BrainOrchestrator::parallelCorticalProcessing() {
    // Process PREVIOUS working memory (t-1) while new input (t) is being encoded
    // This enables pipeline parallelism
    
    if (pipeline_state_.wm_previous.empty()) {
        // First token: nothing to process yet
        return;
    }
    
    // === Wernicke's Area: Semantic Processing ===
    wernicke_->receiveInput(pipeline_state_.wm_previous);
    wernicke_->update(config_.time_step_ms, average_reward_);
    auto semantic_output = wernicke_->getOutputState();
    
    // === Hippocampus: Memory Retrieval ===
    // Retrieves relevant memories based on current semantic content
    hippocampus_->receiveInput(semantic_output);
    hippocampus_->update(config_.time_step_ms, average_reward_);
    auto retrieved_memory = hippocampus_->getOutputState();
    
    // === PFC: Integration with Recurrent Context ===
    // Combine current semantic content with:
    // 1. Retrieved memories from hippocampus
    // 2. Recurrent hidden state (temporal context)
    // 3. Accumulated context buffer
    
    std::vector<float> pfc_input;
    pfc_input.reserve(semantic_output.size() + 
                     retrieved_memory.size() + 
                     pipeline_state_.pfc_hidden_state.size() +
                     pipeline_state_.wm_context.size());
    
    // Concatenate all context sources
    pfc_input.insert(pfc_input.end(), semantic_output.begin(), semantic_output.end());
    pfc_input.insert(pfc_input.end(), retrieved_memory.begin(), retrieved_memory.end());
    
    // Add recurrent state if it exists
    if (!pipeline_state_.pfc_hidden_state.empty()) {
        pfc_input.insert(pfc_input.end(), 
                        pipeline_state_.pfc_hidden_state.begin(), 
                        pipeline_state_.pfc_hidden_state.end());
    }
    
    // Add accumulated context
    if (!pipeline_state_.wm_context.empty()) {
        // Sample from context buffer (don't overwhelm PFC)
        size_t context_samples = std::min(pipeline_state_.wm_context.size(), size_t(256));
        pfc_input.insert(pfc_input.end(), 
                        pipeline_state_.wm_context.begin(), 
                        pipeline_state_.wm_context.begin() + context_samples);
    }
    
    pfc_->receiveInput(pfc_input);
    pfc_->update(config_.time_step_ms, average_reward_);
    
    // Track tokens in pipeline
    pipeline_state_.tokens_in_pipeline++;
}

std::vector<float> BrainOrchestrator::conditionalOutputGeneration() {
    std::vector<float> output;
    
    if (pipeline_state_.tokens_in_pipeline == 0) {
        // No tokens processed yet
        return output;
    }
    
    // Get integrated state from PFC
    auto pfc_integrated = pfc_->getOutputState();
    
    // === Basal Ganglia: Action Selection (Go/No-Go) ===
    // Decides whether to output now or accumulate more context
    basal_ganglia_->receiveInput(pfc_integrated);
    basal_ganglia_->update(config_.time_step_ms, average_reward_);
    
    auto bg_decision = basal_ganglia_->getOutputState();
    float decision_confidence = std::accumulate(
        bg_decision.begin(), bg_decision.end(), 0.0f
    ) / std::max(float(bg_decision.size()), 1.0f);
    
    // Output generation is CONDITIONAL based on:
    // 1. Basal ganglia confidence (learned timing)
    // 2. Maximum pipeline depth (prevent indefinite accumulation)
    bool should_output = (decision_confidence > 0.5f) || 
                        (pipeline_state_.tokens_in_pipeline >= config_.max_pipeline_depth);
    
    if (should_output) {
        // === Broca's Area: Language Production ===
        // Generate output from accumulated PFC context
        
        // Apply top-down bias (disinhibit Broca)
        broca_->applyTopDownBias(0.8f);
        
        broca_->receiveInput(pfc_integrated);
        broca_->update(config_.time_step_ms, average_reward_);
        output = broca_->getOutputState();
        
        // Reset pipeline counter
        pipeline_state_.tokens_in_pipeline = 0;
        tokens_processed_++;
        
        // Store working memory for potential retrieval
        pfc_->setWorkingMemory(pfc_integrated);
        
    } else {
        // Continue accumulating context (No-Go signal)
        broca_->applyTopDownBias(0.0f);  // Maintain inhibition
    }
    
    return output;
}

void BrainOrchestrator::updateRecurrentState() {
    // Update recurrent hidden states for temporal context maintenance

    // PFC hidden state (working memory maintenance)
    auto pfc_state = pfc_->getOutputState();
    if (!pfc_state.empty()) {
        // Exponential moving average of PFC state
        if (pipeline_state_.pfc_hidden_state.empty()) {
            pipeline_state_.pfc_hidden_state = pfc_state;
        } else {
            float alpha = 0.7f;  // Recurrent connection strength
            for (size_t i = 0; i < std::min(pipeline_state_.pfc_hidden_state.size(),
                                            pfc_state.size()); ++i) {
                pipeline_state_.pfc_hidden_state[i] =
                    alpha * pipeline_state_.pfc_hidden_state[i] +
                    (1.0f - alpha) * pfc_state[i];
            }
        }
    }

    // Hippocampus hidden state (episodic memory trace)
    auto hippo_state = hippocampus_->getOutputState();
    if (!hippo_state.empty()) {
        if (pipeline_state_.hippocampus_hidden_state.empty()) {
            pipeline_state_.hippocampus_hidden_state = hippo_state;
        } else {
            float alpha = 0.5f;  // Memory decay rate
            for (size_t i = 0; i < std::min(pipeline_state_.hippocampus_hidden_state.size(),
                                            hippo_state.size()); ++i) {
                pipeline_state_.hippocampus_hidden_state[i] =
                    alpha * pipeline_state_.hippocampus_hidden_state[i] +
                    (1.0f - alpha) * hippo_state[i];
            }
        }
    }
}

// ============================================================================
// BIOLOGICAL SYNCHRONOUS PROCESSING IMPLEMENTATION
// ============================================================================

/**
 * @brief Tick-Tock Synchronous Execution for Biology-First Architecture
 *
 * This mode implements true synchronous execution with explicit phase separation:
 * 1. Integration: All modules compute membrane potential (V_m) from inputs at t-1
 * 2. Firing: All modules generate spikes based on V_m
 * 3. Routing: Spikes propagate through connections (axonal delays)
 * 4. Plasticity: STDP and structural updates (if enabled)
 *
 * Key benefits:
 * - Enables gamma oscillations and feedback loops
 * - Biologically realistic timing
 * - Winner-take-all dynamics work correctly
 * - Prevents race conditions in recurrent networks
 */
std::vector<float> BrainOrchestrator::biologicalSynchronousStep(
    const std::vector<float>& input_embedding,
    int target_token_id,
    GPUDecoder* decoder)
{
    (void)target_token_id;  // Unused in biological mode
    (void)decoder;          // Unused in biological mode

    std::vector<float> output;

    // PHASE 1: INTEGRATION
    // All modules compute membrane potential from t-1 inputs
    // Inputs are: external input + synaptic currents from previous step
    phaseIntegration();

    // PHASE 2: FIRING
    // All modules generate spikes based on V_m computed in Phase 1
    // Synchronization barrier ensures all V_m calculations complete first
    phaseFiring();

    // PHASE 3: ROUTING
    // Spikes travel through axons to destination modules
    // This implements the biological "axonal delay"
    phaseRouting();

    // PHASE 4: PLASTICITY (if enabled)
    // Update synaptic weights based on spike timing
    // This happens in "slow time" relative to spiking
    if (config_.enable_parallel_execution) {
        phasePlasticity();
    }

    // Inject new input for next iteration
    // This goes into the input buffer, will be processed in next integration phase
    if (!input_embedding.empty()) {
        thalamus_->receiveInput(input_embedding);
    }

    // Generate output if Basal Ganglia signals "ready"
    auto bg_state = basal_ganglia_->getOutputState();
    if (!bg_state.empty()) {
        float bg_activity = std::accumulate(bg_state.begin(), bg_state.end(), 0.0f) /
                           static_cast<float>(bg_state.size());

        if (bg_activity > 0.5f) {
            output = broca_->getOutputState();
            tokens_processed_++;
        }
    }

    // Update timers
    total_time_ += config_.time_step_ms;
    time_since_consolidation_ += config_.time_step_ms;

    // Periodic memory consolidation
    if (config_.enable_consolidation &&
        time_since_consolidation_ >= config_.consolidation_interval_ms) {
        consolidateMemory();
        time_since_consolidation_ = 0.0f;
    }

    return output;
}

void BrainOrchestrator::phaseIntegration() {
    // PHASE 1: All modules integrate inputs and compute V_m
    // This happens in parallel on GPU
    // Each module reads from its input buffer (populated in previous routing phase)

    for (auto& [name, module] : module_map_) {
        // Integration kernel: V_m(t) = V_m(t-1)*decay + I_syn + I_ext
        // This is handled internally by the module's update() call
        // We do NOT call update() here - just prepare integration
        // The actual V_m computation will happen in the module's internal state
    }

    // Synchronization barrier (implicit in sequential CPU code)
    // In true parallel implementation, would use cudaDeviceSynchronize()
}

void BrainOrchestrator::phaseFiring() {
    // PHASE 2: All modules check V_m > threshold and generate spikes
    // This must happen AFTER all V_m calculations complete

    for (auto& [name, module] : module_map_) {
        // Update module state (this triggers spiking logic)
        module->update(config_.time_step_ms, average_reward_);
    }

    // Synchronization barrier
    // Ensures all spikes are generated before routing begins
}

void BrainOrchestrator::phaseRouting() {
    // PHASE 3: Route spikes from source modules to destination modules
    // This implements axonal propagation delay

    // Transmit signals through all inter-module connections
    for (auto& connection : connections_) {
        connection->transmit(config_.time_step_ms);
    }

    // Synchronization barrier
    // Ensures all spikes arrive before next integration phase
}

void BrainOrchestrator::phasePlasticity() {
    // PHASE 4: Apply learning rules
    // This uses spike timing from current and recent timesteps

    // Each module applies its internal STDP/plasticity rules
    // In the biological implementation, this would use Dale's Law kernels

    // Distribute reward signal to all modules for reward-modulated learning
    for (auto& [name, module] : module_map_) {
        // Module-internal plasticity happens here
        // (STDP, homeostasis, structural plasticity)
    }

    // Update inter-module connection strengths
    updateConnectionPlasticity(average_reward_);
}
