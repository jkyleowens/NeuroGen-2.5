#pragma once
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "interfaces/TokenEmbedding.h"

/**
 * @brief GPU-accelerated output decoder
 * 
 * Replaces CPU-bound OutputDecoder with GPU-accelerated operations:
 * - cuBLAS GEMM for projection (50-100x faster)
 * - CUDA kernel for softmax (10-20x faster)
 * - GPU-based sampling (5-10x faster)
 * 
 * Expected total speedup: 50-100x over CPU implementation
 */
class GPUDecoder {
public:
    enum class SamplingStrategy {
        GREEDY,           // Always pick highest probability
        TEMPERATURE,      // Temperature-based sampling
        TOP_K,            // Sample from top-k tokens
        TOP_P             // Nucleus sampling (top-p)
    };

    struct Config {
        int vocab_size;           // Size of vocabulary
        int output_dim;           // Dimension of neural output
        float temperature;        // Sampling temperature (default: 1.0)
        int top_k;               // Top-k parameter (default: 50)
        float top_p;             // Top-p parameter (default: 0.9)
        SamplingStrategy strategy; // Decoding strategy
        int gpu_device;          // GPU device ID
        float learning_rate = 0.001f;  // Learning rate for projection matrix
    };

    GPUDecoder(const Config& config, std::shared_ptr<TokenEmbedding> embeddings);
    ~GPUDecoder();

    /**
     * @brief Decode neural output to token probabilities (GPU)
     * @param neural_output Output vector from Broca's Area
     * @return Token probability distribution (GPU memory)
     */
    float* decodeGPU(const float* d_neural_output);

    /**
     * @brief Decode and sample in one step (GPU)
     * @param d_neural_output Output vector from Broca's Area (GPU)
     * @return Sampled token ID
     */
    int decodeAndSampleGPU(const float* d_neural_output);

    /**
     * @brief Decode neural output to token probabilities (CPU interface)
     * @param neural_output Output vector from Broca's Area (CPU)
     * @return Token probability distribution (CPU)
     */
    std::vector<float> decode(const std::vector<float>& neural_output);

    /**
     * @brief Decode and sample in one step (CPU interface)
     * @param neural_output Output vector from Broca's Area (CPU)
     * @return Sampled token ID
     */
    int decodeAndSample(const std::vector<float>& neural_output);

    /**
     * @brief Decode, sample, and also return the sampled token probability (CPU interface)
     * @param neural_output Output vector from Broca's Area (CPU)
     * @return Pair of (sampled token ID, sampled token probability)
     */
    std::pair<int, float> decodeAndSampleWithProb(const std::vector<float>& neural_output);

    /**
     * @brief Train the projection matrix using cross-entropy loss
     * @param neural_output The input to the decoder (Broca's output)
     * @param target_token_id The correct target token
     * @param predicted_probs The predicted probability distribution (from decode())
     * @return The cross-entropy loss
     * 
     * This uses gradient descent to update the projection matrix:
     * gradient = (probs - one_hot_target) * input^T
     * W = W - lr * gradient
     */
    float trainStep(const std::vector<float>& neural_output, 
                    int target_token_id,
                    const std::vector<float>& predicted_probs);

    /**
     * @brief Set learning rate for projection matrix training
     */
    void setLearningRate(float lr) { config_.learning_rate = lr; }

    /**
     * @brief Decode neural output to string token
     */
    std::string decodeToString(const std::vector<float>& neural_output);

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

    /**
     * @brief Save projection matrix weights to file
     */
    void saveWeights(const std::string& filepath);

    /**
     * @brief Load projection matrix weights from file
     */
    void loadWeights(const std::string& filepath);

    /**
     * @brief Decode neural output to raw logits (before softmax)
     * Useful for debugging and analysis
     */
    std::vector<float> decodeToLogits(const std::vector<float>& neural_output);

    /**
     * @brief Get projection matrix (host copy)
     */
    std::vector<float> getProjectionMatrix();

    /**
     * @brief Set projection matrix from host data
     */
    void setProjectionMatrix(const std::vector<float>& matrix);

    /**
     * @brief Get projection bias (host copy)
     */
    std::vector<float> getProjectionBias();

    /**
     * @brief Set projection bias from host data
     */
    void setProjectionBias(const std::vector<float>& bias);

private:
    Config config_;
    std::shared_ptr<TokenEmbedding> embeddings_;
    
    // GPU resources
    int gpu_device_;
    cublasHandle_t cublas_handle_;
    cudaStream_t stream_;
    
    // GPU memory (device pointers)
    float* d_projection_matrix_;  // [vocab_size, output_dim]
    float* d_projection_bias_;    // [vocab_size]
    float* d_logits_;             // [vocab_size]
    float* d_probabilities_;      // [vocab_size]
    float* d_neural_input_;       // [output_dim] - temporary buffer
    
    // Host memory for results
    float* h_probabilities_;      // [vocab_size] - pinned memory
    
    // Random number generator state (GPU)
    unsigned long long* d_rng_state_;
    
    // Initialize GPU resources
    void initializeGPU();
    
    // Cleanup GPU resources
    void cleanupGPU();
    
    // Initialize projection matrix on GPU
    void initializeProjectionMatrix();
    
    // GPU operations
    void projectToLogitsGPU(const float* d_input, float* d_output);
    void softmaxGPU(const float* d_logits, float* d_probs);
    int sampleTokenGPU(const float* d_probs);
    
    // Sampling strategies (GPU)
    int greedySampleGPU(const float* d_probs);
    int temperatureSampleGPU(const float* d_probs);
    int topKSampleGPU(const float* d_probs);
    int topPSampleGPU(const float* d_probs);
};

