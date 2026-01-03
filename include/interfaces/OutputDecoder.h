#pragma once
#include <vector>
#include <string>
#include <memory>
#include "interfaces/TokenEmbedding.h"

/**
 * @brief Output decoder interface for converting neural outputs to text tokens
 * 
 * This class converts the output vectors from Broca's Area into token probabilities
 * and performs sampling/decoding to generate text output.
 */
class OutputDecoder {
public:
    enum class SamplingStrategy {
        GREEDY,           // Always pick highest probability
        TEMPERATURE,      // Temperature-based sampling
        TOP_K,            // Sample from top-k tokens
        TOP_P,            // Nucleus sampling (top-p)
        BEAM_SEARCH       // Beam search decoding
    };

    struct Config {
        int vocab_size;           // Size of vocabulary
        int output_dim;           // Dimension of neural output
        float temperature;        // Sampling temperature (default: 1.0)
        int top_k;               // Top-k parameter (default: 50)
        float top_p;             // Top-p parameter (default: 0.9)
        int beam_width;          // Beam search width (default: 5)
        SamplingStrategy strategy; // Decoding strategy
    };

    OutputDecoder(const Config& config, std::shared_ptr<TokenEmbedding> embeddings);
    ~OutputDecoder() = default;

    /**
     * @brief Decode neural output to token probabilities
     * @param neural_output Output vector from Broca's Area
     * @return Token probability distribution
     */
    std::vector<float> decode(const std::vector<float>& neural_output);

    /**
     * @brief Sample a token from probability distribution
     * @param probabilities Token probabilities
     * @return Sampled token ID
     */
    int sampleToken(const std::vector<float>& probabilities);

    /**
     * @brief Decode and sample in one step
     * @param neural_output Output vector from Broca's Area
     * @return Sampled token ID
     */
    int decodeAndSample(const std::vector<float>& neural_output);

    /**
     * @brief Decode neural output to string token
     * @param neural_output Output vector from Broca's Area
     * @return Decoded token string
     */
    std::string decodeToString(const std::vector<float>& neural_output);

    /**
     * @brief Decode a sequence of neural outputs to string
     * @param neural_outputs Vector of output vectors
     * @return Decoded string sequence
     */
    std::string decodeSequence(const std::vector<std::vector<float>>& neural_outputs);

    /**
     * @brief Beam search decoding
     * @param neural_outputs Sequence of neural outputs
     * @param beam_width Width of beam
     * @return Best decoded sequence
     */
    std::vector<int> beamSearch(const std::vector<std::vector<float>>& neural_outputs,
                                int beam_width);

    /**
     * @brief Set sampling temperature
     */
    void setTemperature(float temperature) { config_.temperature = temperature; }

    /**
     * @brief Set sampling strategy
     */
    void setSamplingStrategy(SamplingStrategy strategy) { config_.strategy = strategy; }

    /**
     * @brief Get current configuration
     */
    const Config& getConfig() const { return config_; }

private:
    Config config_;
    std::shared_ptr<TokenEmbedding> embeddings_;
    
    // Projection layer from neural output to vocabulary logits
    // Simple linear projection: output_dim -> vocab_size
    std::vector<std::vector<float>> projection_matrix_;
    std::vector<float> projection_bias_;
    
    // Initialize projection layer
    void initializeProjection();
    
    // Apply projection to get logits
    std::vector<float> projectToLogits(const std::vector<float>& neural_output);
    
    // Apply softmax to get probabilities
    std::vector<float> softmax(const std::vector<float>& logits);
    
    // Sampling strategies
    int greedySample(const std::vector<float>& probabilities);
    int temperatureSample(const std::vector<float>& probabilities);
    int topKSample(const std::vector<float>& probabilities);
    int topPSample(const std::vector<float>& probabilities);
    
    // Helper: apply temperature scaling to logits
    std::vector<float> applyTemperature(const std::vector<float>& logits, float temperature);
};

