#include "interfaces/TrainingLoop.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <iomanip>

TrainingLoop::TrainingLoop(const Config& config,
                           std::shared_ptr<BrainOrchestrator> brain,
                           std::shared_ptr<TokenEmbedding> embeddings,
                           std::shared_ptr<OutputDecoder> decoder)
    : config_(config),
      brain_(brain),
      embeddings_(embeddings),
      decoder_(decoder) {
    
    // Initialize metrics
    current_metrics_ = {0.0f, 0.0f, 0.0f, 0, 0, 0.0f};
    
    std::cout << "🎓 Initialized TrainingLoop" << std::endl;
}

void TrainingLoop::train() {
    std::cout << "🚀 Starting training for " << config_.max_epochs << " epochs..." << std::endl;
    
    // Load training data
    loadTrainingData();
    
    for (int epoch = 0; epoch < config_.max_epochs; ++epoch) {
        std::cout << "\n📚 Epoch " << (epoch + 1) << "/" << config_.max_epochs << std::endl;
        
        // Shuffle training data (simplified - just process in order for now)
        int num_batches = train_sequences_.size() / config_.batch_size;
        
        for (int batch = 0; batch < num_batches; ++batch) {
            // Get batch
            int start_idx = batch * config_.batch_size;
            int end_idx = std::min(start_idx + config_.batch_size, 
                                   static_cast<int>(train_sequences_.size()));
            
            std::vector<std::vector<std::string>> batch_sequences(
                train_sequences_.begin() + start_idx,
                train_sequences_.begin() + end_idx
            );
            std::vector<std::vector<std::string>> batch_targets(
                train_targets_.begin() + start_idx,
                train_targets_.begin() + end_idx
            );
            
            // Train on batch
            auto metrics = trainBatch(batch_sequences, batch_targets);
            updateMetrics(metrics);
            
            // Print progress
            if (batch % 10 == 0) {
                printProgress(epoch, batch, metrics);
            }
            
            // Validation
            if (config_.enable_validation && 
                batch % config_.validation_interval == 0 && 
                batch > 0) {
                auto val_metrics = validate();
                std::cout << "\n  📊 Validation - Loss: " << val_metrics.loss 
                         << ", Perplexity: " << val_metrics.perplexity 
                         << ", Accuracy: " << val_metrics.accuracy << std::endl;
            }
        }
        
        // Save checkpoint at end of epoch
        saveCheckpoint(epoch, current_metrics_);
    }
    
    std::cout << "\n✓ Training complete!" << std::endl;
}

TrainingLoop::Metrics TrainingLoop::trainBatch(
    const std::vector<std::vector<std::string>>& sequences,
    const std::vector<std::vector<std::string>>& targets) {
    
    Metrics batch_metrics = {0.0f, 0.0f, 0.0f, 0, 0, 0.0f};
    
    for (size_t i = 0; i < sequences.size(); ++i) {
        const auto& sequence = sequences[i];
        const auto& target = targets[i];
        
        float sequence_loss = 0.0f;
        int correct_predictions = 0;
        int total_predictions = 0;
        
        // Process each token in sequence
        for (size_t t = 0; t < sequence.size() && t < target.size(); ++t) {
            // Encode input token
            auto embedding = embeddings_->encode(sequence[t]);
            
            // Run cognitive step
            auto neural_output = brain_->cognitiveStep(embedding);
            
            // Decode to probabilities
            auto probabilities = decoder_->decode(neural_output);
            
            // Get target token ID
            int target_id = embeddings_->getTokenId(target[t]);
            
            // Calculate reward (negative log likelihood)
            float reward = calculateReward(probabilities, target_id);
            sequence_loss += -reward;  // Loss is negative reward
            
            // Distribute reward to brain
            brain_->distributeReward(reward);
            
            // Check if prediction is correct
            int predicted_id = decoder_->sampleToken(probabilities);
            if (predicted_id == target_id) {
                correct_predictions++;
            }
            total_predictions++;
            
            batch_metrics.tokens_processed++;
        }
        
        // Update batch metrics
        batch_metrics.loss += sequence_loss / sequence.size();
        batch_metrics.accuracy += static_cast<float>(correct_predictions) / total_predictions;
        batch_metrics.sequences_processed++;
    }
    
    // Average over batch
    int batch_size = sequences.size();
    if (batch_size > 0) {
        batch_metrics.loss /= batch_size;
        batch_metrics.accuracy /= batch_size;
        batch_metrics.perplexity = std::exp(batch_metrics.loss);
    }
    
    // Get average reward from brain
    auto brain_stats = brain_->getStats();
    batch_metrics.average_reward = brain_stats.average_reward;
    
    return batch_metrics;
}

TrainingLoop::Metrics TrainingLoop::validate() {
    Metrics val_metrics = {0.0f, 0.0f, 0.0f, 0, 0, 0.0f};
    
    // Similar to trainBatch but without reward distribution
    for (size_t i = 0; i < val_sequences_.size(); ++i) {
        const auto& sequence = val_sequences_[i];
        const auto& target = val_targets_[i];
        
        float sequence_loss = 0.0f;
        int correct_predictions = 0;
        int total_predictions = 0;
        
        for (size_t t = 0; t < sequence.size() && t < target.size(); ++t) {
            auto embedding = embeddings_->encode(sequence[t]);
            auto neural_output = brain_->cognitiveStep(embedding);
            auto probabilities = decoder_->decode(neural_output);
            
            int target_id = embeddings_->getTokenId(target[t]);
            float reward = calculateReward(probabilities, target_id);
            sequence_loss += -reward;
            
            int predicted_id = decoder_->sampleToken(probabilities);
            if (predicted_id == target_id) {
                correct_predictions++;
            }
            total_predictions++;
        }
        
        val_metrics.loss += sequence_loss / sequence.size();
        val_metrics.accuracy += static_cast<float>(correct_predictions) / total_predictions;
        val_metrics.sequences_processed++;
    }
    
    // Average
    if (val_metrics.sequences_processed > 0) {
        val_metrics.loss /= val_metrics.sequences_processed;
        val_metrics.accuracy /= val_metrics.sequences_processed;
        val_metrics.perplexity = std::exp(val_metrics.loss);
    }
    
    return val_metrics;
}

std::string TrainingLoop::generate(const std::string& prompt, int max_length) {
    std::cout << "\n💭 Generating from prompt: \"" << prompt << "\"" << std::endl;
    
    // Tokenize prompt (simple whitespace split)
    std::istringstream iss(prompt);
    std::vector<std::string> prompt_tokens;
    std::string token;
    while (iss >> token) {
        prompt_tokens.push_back(token);
    }
    
    std::string generated = prompt;
    
    // Process prompt tokens
    for (const auto& tok : prompt_tokens) {
        auto embedding = embeddings_->encode(tok);
        brain_->cognitiveStep(embedding);
    }
    
    // Generate new tokens
    for (int i = 0; i < max_length; ++i) {
        // Get last generated token or use BOS
        std::string last_token = prompt_tokens.empty() ? "<BOS>" : prompt_tokens.back();
        auto embedding = embeddings_->encode(last_token);
        
        // Run cognitive step
        auto neural_output = brain_->cognitiveStep(embedding);
        
        if (neural_output.empty()) {
            continue;  // Brain not ready to output yet
        }
        
        // Decode to token
        std::string new_token = decoder_->decodeToString(neural_output);
        
        // Check for EOS
        if (new_token == "<EOS>") {
            break;
        }
        
        // Add to generated text
        generated += " " + new_token;
        prompt_tokens.push_back(new_token);
    }
    
    return generated;
}

float TrainingLoop::calculateReward(const std::vector<float>& predicted_probs, 
                                   int target_token) {
    // Reward = log probability of correct token (negative loss)
    if (target_token < 0 || target_token >= static_cast<int>(predicted_probs.size())) {
        return -10.0f;  // Large negative reward for invalid target
    }
    
    float prob = predicted_probs[target_token];
    prob = std::max(1e-8f, prob);  // Avoid log(0)
    
    return std::log(prob);  // Positive if prob > exp(-1), negative otherwise
}

void TrainingLoop::loadTrainingData() {
    std::cout << "📂 Loading training data..." << std::endl;
    
    if (!config_.train_data_path.empty()) {
        train_sequences_ = loadSequences(config_.train_data_path);
        // For autoregressive training, targets are input shifted by 1
        train_targets_ = train_sequences_;
    } else {
        // Demo data if no file provided
        train_sequences_ = {
            {"The", "cat", "sat", "on", "the", "mat"},
            {"Hello", "world", "how", "are", "you"},
            {"Neural", "networks", "are", "interesting"}
        };
        train_targets_ = train_sequences_;
    }
    
    if (config_.enable_validation && !config_.val_data_path.empty()) {
        val_sequences_ = loadSequences(config_.val_data_path);
        val_targets_ = val_sequences_;
    }
    
    std::cout << "✓ Loaded " << train_sequences_.size() << " training sequences" << std::endl;
}

std::vector<std::vector<std::string>> TrainingLoop::loadSequences(
    const std::string& filepath) {
    
    std::vector<std::vector<std::string>> sequences;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "⚠️  Could not open file: " << filepath << std::endl;
        return sequences;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string token;
        
        while (iss >> token) {
            tokens.push_back(token);
        }
        
        if (!tokens.empty()) {
            sequences.push_back(tokens);
        }
    }
    
    file.close();
    return sequences;
}

void TrainingLoop::updateMetrics(const Metrics& batch_metrics) {
    // Running average
    float alpha = 0.1f;
    current_metrics_.loss = (1 - alpha) * current_metrics_.loss + alpha * batch_metrics.loss;
    current_metrics_.perplexity = std::exp(current_metrics_.loss);
    current_metrics_.accuracy = (1 - alpha) * current_metrics_.accuracy + alpha * batch_metrics.accuracy;
    current_metrics_.tokens_processed += batch_metrics.tokens_processed;
    current_metrics_.sequences_processed += batch_metrics.sequences_processed;
    current_metrics_.average_reward = (1 - alpha) * current_metrics_.average_reward + 
                                     alpha * batch_metrics.average_reward;
}

void TrainingLoop::printProgress(int epoch, int batch, const Metrics& metrics) {
    std::cout << "\r  Batch " << std::setw(4) << batch 
              << " | Loss: " << std::fixed << std::setprecision(4) << metrics.loss
              << " | PPL: " << std::setprecision(2) << metrics.perplexity
              << " | Acc: " << std::setprecision(3) << metrics.accuracy
              << " | Reward: " << std::setprecision(4) << metrics.average_reward
              << std::flush;
}

void TrainingLoop::saveCheckpoint(int epoch, const Metrics& metrics) {
    if (config_.checkpoint_dir.empty()) {
        return;
    }

    namespace fs = std::filesystem;
    fs::path checkpoint_root(config_.checkpoint_dir);
    std::error_code ec;
    fs::create_directories(checkpoint_root, ec);
    if (ec && !fs::exists(checkpoint_root)) {
        std::cerr << "⚠️  Unable to create checkpoint directory: "
                  << checkpoint_root << " (" << ec.message() << ")" << std::endl;
        return;
    }

    const std::string epoch_suffix = "checkpoint_epoch_" + std::to_string(epoch);

    fs::path metrics_path = checkpoint_root / (epoch_suffix + ".txt");
    std::ofstream file(metrics_path);
    if (file.is_open()) {
        file << "Epoch: " << epoch << "\n";
        file << "Loss: " << metrics.loss << "\n";
        file << "Perplexity: " << metrics.perplexity << "\n";
        file << "Accuracy: " << metrics.accuracy << "\n";
        file << "Tokens: " << metrics.tokens_processed << "\n";
        file << "Sequences: " << metrics.sequences_processed << "\n";
        file << "Reward: " << metrics.average_reward << "\n";
        file.close();
        
        std::cout << "\n📝 Metrics checkpoint saved: " << metrics_path << std::endl;
    } else {
        std::cerr << "⚠️  Failed to write checkpoint metrics: " << metrics_path << std::endl;
    }

    if (brain_) {
        fs::path binary_path = checkpoint_root / (epoch_suffix + ".ngchk");
        if (!brain_->saveCheckpoint(binary_path.string())) {
            std::cerr << "⚠️  Failed to save binary checkpoint: " << binary_path << std::endl;
        }
    }
}

