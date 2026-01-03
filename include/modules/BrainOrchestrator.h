#pragma once

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <iostream>

#include "modules/CorticalModule.h"
#include "modules/InterModuleConnection.h"
#include "modules/FeedbackMatrix.h"
#include "interfaces/GPUDecoder.h"
#include "persistence/NetworkSnapshot.h"

class BrainOrchestrator {
public:
    enum class ProcessingMode {
        SEQUENTIAL,
        PIPELINED,
        BIOLOGICAL_SYNC  // Tick-tock synchronous execution for gamma oscillations
    };

    struct Config {
        int gpu_device_id = 0;
        float time_step_ms = 1.0f;
        bool enable_parallel_execution = false;
        bool enable_consolidation = true;
        float consolidation_interval_ms = 10000.0f;
        ProcessingMode processing_mode = ProcessingMode::SEQUENTIAL;
        int max_pipeline_depth = 5;
    };

    // Nested struct for detailed stats
    struct ModuleStats {
        float activity_level;
        float dopamine_level;
        float serotonin_level;
    };

    // Deprecated phase enum (kept for compatibility)
    enum class CognitivePhase { SENSATION, PERCEPTION, INTEGRATION, SELECTION, ACTION };

    struct Stats {
        float total_time_ms;
        int cognitive_cycles;
        int tokens_processed;
        float average_reward;
        CognitivePhase current_phase; 
        std::unordered_map<std::string, ModuleStats> module_stats; // Detailed per-module stats
    };

    explicit BrainOrchestrator(const Config& config);
    ~BrainOrchestrator();

    void initializeModules();
    void createConnectome();
    
    // Core cognitive step
    std::vector<float> cognitiveStep(
        const std::vector<float>& input_embedding, 
        int target_token_id = -1, 
        GPUDecoder* decoder = nullptr
    );

    // Pipelined cognitive step
    std::vector<float> pipelinedCognitiveStep(
        const std::vector<float>& input_embedding,
        int target_token_id = -1,
        GPUDecoder* decoder = nullptr
    );

    // Biological synchronous step (Tick-Tock)
    std::vector<float> biologicalSynchronousStep(
        const std::vector<float>& input_embedding,
        int target_token_id = -1,
        GPUDecoder* decoder = nullptr
    );

    // Helper to access global workspace (concatenated state)
    std::vector<float> getGlobalWorkspace();

    void reset();
    std::vector<float> getBrocaOutput();
    
    // Neuromodulation
    void modulateGlobalState(float dopamine, float serotonin, float norepinephrine);
    
    // FIXED: Moved distributeReward to public so TrainingLoop can call it
    void distributeReward(float reward);
    
    bool saveCheckpoint(const std::string& file_path) const;
    bool loadCheckpoint(const std::string& file_path);
    
    Stats getStats() const;
    void setProcessingMode(ProcessingMode mode);

    persistence::BrainSnapshot captureSnapshot() const;
    persistence::BrainSnapshot captureMetadataSnapshot() const; // Lightweight capture

private:
    Config config_;
    
    // Module ownership
    std::unique_ptr<CorticalModule> thalamus_;
    std::unique_ptr<CorticalModule> wernicke_;
    std::unique_ptr<CorticalModule> broca_;
    std::unique_ptr<CorticalModule> hippocampus_;
    std::unique_ptr<CorticalModule> pfc_;
    std::unique_ptr<CorticalModule> basal_ganglia_;
    
    std::unordered_map<std::string, CorticalModule*> module_map_;
    std::vector<std::unique_ptr<InterModuleConnection>> connections_;
    
    // Feedback Matrices
    std::unique_ptr<FeedbackMatrix> feedback_output_to_broca_;
    std::unique_ptr<FeedbackMatrix> feedback_broca_to_wernicke_;

    // State tracking
    std::vector<float> pending_input_;
    bool should_generate_output_;
    
    // Phases
    CognitivePhase current_phase_;
    float phase_timer_;
    
    float total_time_;
    float time_since_consolidation_;
    int cognitive_cycles_;
    int tokens_processed_;
    float average_reward_;
    
    // Processing Mode
    ProcessingMode processing_mode_;
    
    // Pipeline state
    struct PipelineState {
        std::vector<float> wm_current;
        std::vector<float> wm_previous;
        std::vector<float> wm_context;
        std::vector<float> pfc_hidden_state;
        std::vector<float> hippocampus_hidden_state;
        int tokens_in_pipeline = 0;
        float accumulated_processing_time = 0.0f;
    } pipeline_state_;

    // Internal helpers
    void fastInputEncoding(const std::vector<float>& input, const std::vector<float>& prediction);
    void parallelCorticalProcessing();
    std::vector<float> conditionalOutputGeneration();
    void updateRecurrentState();
    void consolidateMemory();
    void routeSignals();

    // Biological synchronous helpers
    void phaseIntegration();      // Phase 1: All modules compute V_m
    void phaseFiring();           // Phase 2: All modules generate spikes
    void phaseRouting();          // Phase 3: Spikes travel via axons
    void phasePlasticity();       // Phase 4: STDP and structural updates
    
    void updateConnectionPlasticity(float reward);
    
    CorticalModule* getModule(const std::string& name);
    
    void advancePhase();
    float getPhaseDuration(CognitivePhase phase) const;
    void executeSensationPhase(const std::vector<float>& input);
    void executePerceptionPhase();
    void executeIntegrationPhase();
    void executeSelectionPhase();
    std::vector<float> executeActionPhase();
};