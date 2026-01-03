#include "interfaces/OutputDecoder.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>

OutputDecoder::OutputDecoder(const Config& config, 
                             std::shared_ptr<TokenEmbedding> embeddings)
    : config_(config), embeddings_(embeddings) {
    
    initializeProjection();
    
    std::cout << "🔊 Initialized OutputDecoder with vocab_size=" 
              << config_.vocab_size << ", output_dim=" 
              << config_.output_dim << std::endl;
}

void OutputDecoder::initializeProjection() {
    // Direct Character Output: No projection matrix needed.
    // We map neural outputs directly to character probabilities.
    std::cout << "🔤 Initialized Direct Character Output (No Projection Matrix)" << std::endl;
    
    // Clear projection matrix to save memory
    projection_matrix_.clear();
    projection_bias_.clear();
}

std::vector<float> OutputDecoder::decode(const std::vector<float>& neural_output) {
    // Direct mapping: Neural output indices 0-255 correspond to characters
    std::vector<float> logits(256);
    
    for (int i = 0; i < 256; ++i) {
        if (i < static_cast<int>(neural_output.size())) {
            // Direct pass-through of neural activity to logit
            logits[i] = neural_output[i];
        } else {
            logits[i] = -100.0f; // effectively zero probability
        }
    }
    
    // Apply softmax to get probabilities
    return softmax(logits);
}

std::vector<float> OutputDecoder::projectToLogits(const std::vector<float>& neural_output) {
    // This function is now redundant but kept for interface compatibility
    // It just extracts the first 256 values
    std::vector<float> logits(256);
    for (int i = 0; i < 256; ++i) {
        if (i < static_cast<int>(neural_output.size())) {
            logits[i] = neural_output[i];
        } else {
            logits[i] = 0.0f;
        }
    }
    return logits;
}

std::vector<float> OutputDecoder::softmax(const std::vector<float>& logits) {
    std::vector<float> probabilities(logits.size());
    
    // Find max for numerical stability
    float max_logit = *std::max_element(logits.begin(), logits.end());
    
    // Compute exp and sum
    float sum_exp = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = std::exp(logits[i] - max_logit);
        // Clamp to prevent NaNs or Infs if any weirdness occurs
        if (std::isnan(probabilities[i]) || std::isinf(probabilities[i])) {
            probabilities[i] = 0.0f;
        }
        sum_exp += probabilities[i];
    }
    
    // Normalize
    // If sum_exp is effectively zero, use uniform distribution
    if (sum_exp < 1e-9f) {
        float uniform = 1.0f / logits.size();
        std::fill(probabilities.begin(), probabilities.end(), uniform);
    } else {
        for (size_t i = 0; i < probabilities.size(); ++i) {
            probabilities[i] /= sum_exp;
        }
    }
    
    return probabilities;
}

int OutputDecoder::sampleToken(const std::vector<float>& probabilities) {
    switch (config_.strategy) {
        case SamplingStrategy::GREEDY:
            return greedySample(probabilities);
        case SamplingStrategy::TEMPERATURE:
            return temperatureSample(probabilities);
        case SamplingStrategy::TOP_K:
            return topKSample(probabilities);
        case SamplingStrategy::TOP_P:
            return topPSample(probabilities);
        default:
            return greedySample(probabilities);
    }
}

int OutputDecoder::decodeAndSample(const std::vector<float>& neural_output) {
    auto probabilities = decode(neural_output);
    return sampleToken(probabilities);
}

std::string OutputDecoder::decodeToString(const std::vector<float>& neural_output) {
    int token_id = decodeAndSample(neural_output);
    return embeddings_->getToken(token_id);
}

std::string OutputDecoder::decodeSequence(const std::vector<std::vector<float>>& neural_outputs) {
    std::string result;
    
    for (const auto& output : neural_outputs) {
        std::string token = decodeToString(output);
        if (!result.empty()) {
            result += " ";
        }
        result += token;
    }
    
    return result;
}

int OutputDecoder::greedySample(const std::vector<float>& probabilities) {
    // Return index of maximum probability
    return std::distance(probabilities.begin(), 
                        std::max_element(probabilities.begin(), probabilities.end()));
}

int OutputDecoder::temperatureSample(const std::vector<float>& probabilities) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    // Create discrete distribution from probabilities
    std::discrete_distribution<int> dist(probabilities.begin(), probabilities.end());
    
    return dist(gen);
}

int OutputDecoder::topKSample(const std::vector<float>& probabilities) {
    // Create pairs of (probability, index)
    std::vector<std::pair<float, int>> prob_index;
    prob_index.reserve(probabilities.size());
    
    for (size_t i = 0; i < probabilities.size(); ++i) {
        prob_index.push_back({probabilities[i], static_cast<int>(i)});
    }
    
    // Sort by probability (descending)
    std::sort(prob_index.begin(), prob_index.end(), 
             [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Keep only top-k
    int k = std::min(config_.top_k, static_cast<int>(prob_index.size()));
    prob_index.resize(k);
    
    // Renormalize
    float sum = 0.0f;
    for (const auto& p : prob_index) {
        sum += p.first;
    }
    
    std::vector<float> top_k_probs;
    top_k_probs.reserve(k);
    for (const auto& p : prob_index) {
        top_k_probs.push_back(p.first / sum);
    }
    
    // Sample from top-k
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::discrete_distribution<int> dist(top_k_probs.begin(), top_k_probs.end());
    
    int sampled_idx = dist(gen);
    return prob_index[sampled_idx].second;
}

int OutputDecoder::topPSample(const std::vector<float>& probabilities) {
    // Create pairs of (probability, index)
    std::vector<std::pair<float, int>> prob_index;
    prob_index.reserve(probabilities.size());
    
    for (size_t i = 0; i < probabilities.size(); ++i) {
        prob_index.push_back({probabilities[i], static_cast<int>(i)});
    }
    
    // Sort by probability (descending)
    std::sort(prob_index.begin(), prob_index.end(), 
             [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Keep tokens until cumulative probability exceeds top_p
    float cumulative = 0.0f;
    size_t cutoff = 0;
    
    for (size_t i = 0; i < prob_index.size(); ++i) {
        cumulative += prob_index[i].first;
        cutoff = i + 1;
        if (cumulative >= config_.top_p) {
            break;
        }
    }
    
    prob_index.resize(cutoff);
    
    // Renormalize
    std::vector<float> nucleus_probs;
    nucleus_probs.reserve(cutoff);
    for (const auto& p : prob_index) {
        nucleus_probs.push_back(p.first / cumulative);
    }
    
    // Sample from nucleus
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::discrete_distribution<int> dist(nucleus_probs.begin(), nucleus_probs.end());
    
    int sampled_idx = dist(gen);
    return prob_index[sampled_idx].second;
}

std::vector<float> OutputDecoder::applyTemperature(const std::vector<float>& logits, 
                                                   float temperature) {
    std::vector<float> scaled_logits(logits.size());
    
    for (size_t i = 0; i < logits.size(); ++i) {
        scaled_logits[i] = logits[i] / temperature;
    }
    
    return scaled_logits;
}

std::vector<int> OutputDecoder::beamSearch(
    const std::vector<std::vector<float>>& neural_outputs, int beam_width) {
    (void)beam_width; // Unused parameter
    
    // Simplified beam search implementation
    // In practice, this would maintain multiple hypotheses
    
    std::vector<int> best_sequence;
    best_sequence.reserve(neural_outputs.size());
    
    for (const auto& output : neural_outputs) {
        auto probabilities = decode(output);
        int token_id = greedySample(probabilities);  // Simplified
        best_sequence.push_back(token_id);
    }
    
    return best_sequence;
}
