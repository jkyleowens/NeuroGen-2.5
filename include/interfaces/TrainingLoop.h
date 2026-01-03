#pragma once
#include <vector>
#include <string>
#include <memory>
#include "modules/BrainOrchestrator.h"
#include "interfaces/TokenEmbedding.h"
#include "interfaces/OutputDecoder.h"

/**
 * @brief Training loop interface for the modular brain NLP system
 * 
 * Handles sequence-to-sequence training, reward calculation, and metrics tracking.
 */
class TrainingLoop {
public:
    struct Config {
        int max_epochs;
        int batch_size;
        float learning_rate;
        int sequence_length;
        bool enable_validation;
        int validation_interval;  // Check validation every N batches
        float reward_discount;    // Discount factor for reward calculation
        std::string train_data_path;
        std::string val_data_path;
        std::string checkpoint_dir;
    };

    struct Metrics {
        float loss;
        float perplexity;
        float accuracy;
        int tokens_processed;
        int sequences_processed;
        float average_reward;
    };

    TrainingLoop(const Config& config,
                 std::shared_ptr<BrainOrchestrator> brain,
                 std::shared_ptr<TokenEmbedding> embeddings,
                 std::shared_ptr<OutputDecoder> decoder);
    ~TrainingLoop() = default;

    /**
     * @brief Run training for specified number of epochs
     */
    void train();

    /**
     * @brief Train on a single batch
     * @param sequences Batch of input sequences
     * @param targets Batch of target sequences
     * @return Metrics for this batch
     */
    Metrics trainBatch(const std::vector<std::vector<std::string>>& sequences,
                       const std::vector<std::vector<std::string>>& targets);

    /**
     * @brief Validate on validation set
     * @return Validation metrics
     */
    Metrics validate();

    /**
     * @brief Generate text from prompt
     * @param prompt Initial prompt text
     * @param max_length Maximum generation length
     * @return Generated text
     */
    std::string generate(const std::string& prompt, int max_length);

    /**
     * @brief Calculate reward signal from prediction error
     * @param predicted_probs Predicted token probabilities
     * @param target_token Target token ID
     * @return Reward signal (negative log likelihood)
     */
    float calculateReward(const std::vector<float>& predicted_probs, int target_token);

    /**
     * @brief Load training data
     */
    void loadTrainingData();

    /**
     * @brief Save checkpoint
     */
    void saveCheckpoint(int epoch, const Metrics& metrics);

    /**
     * @brief Get current metrics
     */
    const Metrics& getMetrics() const { return current_metrics_; }

private:
    Config config_;
    std::shared_ptr<BrainOrchestrator> brain_;
    std::shared_ptr<TokenEmbedding> embeddings_;
    std::shared_ptr<OutputDecoder> decoder_;
    
    // Training data
    std::vector<std::vector<std::string>> train_sequences_;
    std::vector<std::vector<std::string>> train_targets_;
    std::vector<std::vector<std::string>> val_sequences_;
    std::vector<std::vector<std::string>> val_targets_;
    
    // Metrics tracking
    Metrics current_metrics_;
    std::vector<Metrics> epoch_metrics_;
    
    // Helper functions
    std::vector<std::vector<std::string>> loadSequences(const std::string& filepath);
    void updateMetrics(const Metrics& batch_metrics);
    void printProgress(int epoch, int batch, const Metrics& metrics);
};

