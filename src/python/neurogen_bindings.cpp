/**
 * Python Bindings for NeuroGen 2.0
 * 
 * This file provides Python bindings using pybind11 to allow the
 * train_slimpajama.py script to interface with the C++ NeuroGen engine.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <fstream>
#include "modules/BrainOrchestrator.h"
#include "interfaces/TokenEmbedding.h"
#include "interfaces/OutputDecoder.h"
#include "interfaces/GPUDecoder.h"
#include <cuda_runtime.h>

namespace py = pybind11;

/**
 * Python-facing wrapper for the NeuroGen model
 */
class NeuroGenModel {
public:
    NeuroGenModel(int vocab_size, int embedding_dim, int gpu_device) 
        : vocab_size_(vocab_size), 
          embedding_dim_(embedding_dim),
          gpu_device_(gpu_device) {
        
        // Verify GPU is available
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            throw std::runtime_error("No CUDA-capable GPU found!");
        }
        
        if (gpu_device_ >= device_count) {
            throw std::runtime_error("Invalid GPU device ID: " + std::to_string(gpu_device_));
        }
        
        // Set GPU device
        cudaSetDevice(gpu_device_);
        
        // Print GPU info
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, gpu_device_);
        std::cout << "🎮 Using GPU: " << prop.name 
                  << " (Device " << gpu_device_ << ")" << std::endl;
        std::cout << "   Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "   Total Memory: " << (prop.totalGlobalMem / (1024*1024*1024.0)) << " GB" << std::endl;
        
        // Initialize Brain Orchestrator
        BrainOrchestrator::Config config;
        config.gpu_device_id = gpu_device_;
        config.time_step_ms = 1.0f;
        config.enable_parallel_execution = true;
        config.enable_consolidation = false; // Disable for training speed
        config.processing_mode = BrainOrchestrator::ProcessingMode::PIPELINED; // Enable pipelined mode for efficiency
        config.max_pipeline_depth = 8; // Accumulate up to 8 tokens of context
        
        brain_ = std::make_unique<BrainOrchestrator>(config);
        brain_->initializeModules();
        brain_->createConnectome();
        
        // Initialize embedding and decoder
        TokenEmbedding::Config embed_config;
        embed_config.vocab_size = 256; // Fixed for Direct Character Input
        embed_config.embedding_dim = 256; // Fixed for Direct Character Input
        embed_config.use_random_init = false; // Not needed for fixed mapping
        embed_config.normalization = 1.0f;
        embed_config.vocab_file = "";
        
        embedding_ = std::make_shared<TokenEmbedding>(embed_config);
        embedding_->initialize();
        
        // Use GPU-accelerated decoder for 50-100x speedup!
        GPUDecoder::Config gpu_decoder_config;
        gpu_decoder_config.vocab_size = 256; // Fixed for Direct Character Output
        gpu_decoder_config.output_dim = 256; // Broca output size (Direct Character Output)
        gpu_decoder_config.temperature = 1.0f;
        gpu_decoder_config.top_k = 50;
        gpu_decoder_config.top_p = 0.9f;
        gpu_decoder_config.strategy = GPUDecoder::SamplingStrategy::GREEDY; // Use GREEDY for training (deterministic, measurable accuracy)
        gpu_decoder_config.gpu_device = gpu_device_;
        
        gpu_decoder_ = std::make_shared<GPUDecoder>(gpu_decoder_config, embedding_);
        
        // Create a second decoder for inference with top-k sampling
        GPUDecoder::Config inference_decoder_config = gpu_decoder_config;
        inference_decoder_config.strategy = GPUDecoder::SamplingStrategy::TOP_K;
        inference_decoder_ = std::make_shared<GPUDecoder>(inference_decoder_config, embedding_);
        
        std::cout << "✅ NeuroGen model initialized with:" << std::endl;
        std::cout << "   Vocab Size: 256 (Direct Character)" << std::endl;
        std::cout << "   Embedding Dim: 256 (Direct Character)" << std::endl;
        std::cout << "   GPU Device: " << gpu_device_ << std::endl;
        std::cout << "   🎯 Memory Optimized: GTX 1650 (4GB) configuration" << std::endl;
        std::cout << "   📊 Training Decoder: GREEDY (deterministic)" << std::endl;
        std::cout << "   🎲 Inference Decoder: TOP_K=50 (creative)" << std::endl;
    }

    std::shared_ptr<TokenEmbedding> getEmbedding() const {
        return embedding_;
    }

    std::shared_ptr<GPUDecoder> getGPUDecoder() const {
        return gpu_decoder_;
    }

    BrainOrchestrator* getBrain() { return brain_.get(); }
    TokenEmbedding* getEmbeddingPtr() { return embedding_.get(); }
    GPUDecoder* getGPUDecoderPtr() { return gpu_decoder_.get(); }

    /**
     * Train on a single step (predict exactly ONE next token)
     * Uses GREEDY decoding for deterministic, measurable accuracy
     * 
     * The model's internal working memory (PFC, Hippocampus, pipeline state)
     * maintains temporal context automatically between calls.
     * 
     * Training updates:
     * 1. SNN weights via STDP + reward modulation
     * 2. Decoder projection matrix via gradient descent
     * 
     * @param input_ids: List of input token IDs (full context sequence)
     * @param target_ids: List of target token IDs (we use the last one)
     * @return: Tuple of (loss, accuracy, predicted_token_id)
     */
    std::tuple<float, float, int> train_step(
        const std::vector<int>& input_ids, 
        const std::vector<int>& target_ids) {
        
        if (input_ids.empty() || target_ids.empty()) {
            throw std::runtime_error("Input and target must be non-empty");
        }
        
        // Process each token in sequence through the brain
        // Internal working memory accumulates context automatically
        for (size_t i = 0; i < input_ids.size(); ++i) {
            std::vector<float> embedded = embedding_->encodeById(input_ids[i]);
            brain_->cognitiveStep(embedded);
        }

        // Get output and decode to full probability distribution
        std::vector<float> brain_output = brain_->getBrocaOutput();
        std::vector<float> probs = gpu_decoder_->decode(brain_output);
        
        // Sample token and get its probability
        auto token_and_prob = gpu_decoder_->decodeAndSampleWithProb(brain_output);
        int predicted_token = token_and_prob.first;
        float predicted_prob = token_and_prob.second;

        int target_token = target_ids.back();

        // Compute loss
        const float eps = 1e-8f;
        float nll = -std::log(std::max(predicted_prob, eps));
        float loss = nll;
        float accuracy = (predicted_token == target_token) ? 1.0f : 0.0f;

        // === CRITICAL: Train the decoder projection matrix ===
        // This updates W to better map neural activations to correct tokens
        gpu_decoder_->trainStep(brain_output, target_token, probs);

        // Apply reward signal for SNN learning (STDP + neuromodulation)
        // Use smaller reward magnitude to prevent weight explosion
        float reward = accuracy > 0.5f ? 0.1f : -0.1f;
        brain_->distributeReward(reward);
        brain_->modulateGlobalState(reward * 0.5f, 0.3f, 0.3f);

        return std::make_tuple(loss, accuracy, predicted_token);
    }
    
    /**
     * Generate text from a prompt (uses TOP_K sampling for creative output)
     * @param prompt_ids: List of prompt token IDs
     * @param max_length: Maximum generation length
     * @return: List of generated token IDs
     */
    std::vector<int> generate(const std::vector<int>& prompt_ids, int max_length) {
        std::vector<int> generated = prompt_ids;
        
        // Process prompt
        for (int token_id : prompt_ids) {
            std::vector<float> embedded = embedding_->encodeById(token_id);
            brain_->cognitiveStep(embedded);
        }
        
        // Generate new tokens using inference decoder (TOP_K for creativity)
        for (int i = 0; i < max_length; ++i) {
            std::vector<float> brain_output = brain_->getBrocaOutput();
            int next_token = inference_decoder_->decodeAndSample(brain_output);
            
            generated.push_back(next_token);
            
            // Feed generated token back as input
            std::vector<float> embedded = embedding_->encodeById(next_token);
            brain_->cognitiveStep(embedded);
            
            // Stop on EOS token (assuming token 50256 is EOS for GPT-2)
            if (next_token == 50256) break;
        }
        
        return generated;
    }
    
    /**
     * Save checkpoint (includes brain state, embeddings, and decoder)
     */
    void save_checkpoint(const std::string& path) {
        // Save brain orchestrator state
        if (!brain_->saveCheckpoint(path)) {
            throw std::runtime_error("Failed to save brain checkpoint");
        }
        
        // Save embeddings
        std::string embedding_path = path + ".embeddings";
        embedding_->saveEmbeddings(embedding_path);
        
        // Save decoder projection matrix
        std::string decoder_path = path + ".decoder";
        gpu_decoder_->saveWeights(decoder_path);
        
        std::cout << "✅ Checkpoint saved: " << path << std::endl;
    }
    
    /**
     * Load checkpoint (includes brain state, embeddings, and decoder)
     */
    void load_checkpoint(const std::string& path) {
        // Load brain orchestrator state
        if (!brain_->loadCheckpoint(path)) {
            throw std::runtime_error("Failed to load brain checkpoint");
        }
        
        // Load embeddings if they exist
        std::string embedding_path = path + ".embeddings";
        std::ifstream emb_check(embedding_path);
        if (emb_check.good()) {
            emb_check.close();
            embedding_->loadEmbeddings(embedding_path);
        } else {
            std::cout << "⚠️  No embedding file found, using current embeddings" << std::endl;
        }
        
        // Load decoder projection matrix if it exists
        std::string decoder_path = path + ".decoder";
        std::ifstream dec_check(decoder_path);
        if (dec_check.good()) {
            dec_check.close();
            gpu_decoder_->loadWeights(decoder_path);
        } else {
            std::cout << "⚠️  No decoder file found, using current weights" << std::endl;
        }
        
        std::cout << "✅ Checkpoint loaded: " << path << std::endl;
    }

    /**
     * Get brain statistics
     */
    py::dict get_statistics() {
        auto stats = brain_->getStats();
        
        py::dict result;
        result["cognitive_cycles"] = stats.cognitive_cycles;
        result["tokens_processed"] = stats.tokens_processed;
        result["average_reward"] = stats.average_reward;
        result["total_time_ms"] = stats.total_time_ms;
        
        // Module activities
        py::dict module_stats;
        for (const auto& [name, mod_stat] : stats.module_stats) {
            py::dict mod_dict;
            mod_dict["activity_level"] = mod_stat.activity_level;
            mod_dict["dopamine_level"] = mod_stat.dopamine_level;
            mod_dict["serotonin_level"] = mod_stat.serotonin_level;
            module_stats[name.c_str()] = mod_dict;
        }
        result["modules"] = module_stats;
        
        return result;
    }

    /**
     * Get diagnostic info about neural activity and decoder
     * Useful for debugging mode collapse and dead neurons
     */
    py::dict get_diagnostics() {
        py::dict result;
        
        // Get Broca output
        std::vector<float> broca_output = brain_->getBrocaOutput();
        
        // Compute sparsity and statistics
        int active_count = 0;
        float sum = 0.0f;
        float max_val = -1e30f;
        float min_val = 1e30f;
        
        for (float v : broca_output) {
            if (v > 0.01f) active_count++;
            sum += v;
            max_val = std::max(max_val, v);
            min_val = std::min(min_val, v);
        }
        
        float sparsity = 1.0f - (float)active_count / broca_output.size();
        float mean = sum / broca_output.size();
        
        result["broca_output_size"] = broca_output.size();
        result["broca_active_neurons"] = active_count;
        result["broca_sparsity"] = sparsity;
        result["broca_mean"] = mean;
        result["broca_max"] = max_val;
        result["broca_min"] = min_val;
        
        // Get logit statistics from decoder
        std::vector<float> logits = gpu_decoder_->decodeToLogits(broca_output);
        
        float logit_sum = 0.0f;
        float logit_max = -1e30f;
        float logit_min = 1e30f;
        float logit_sq_sum = 0.0f;
        
        for (float l : logits) {
            logit_sum += l;
            logit_sq_sum += l * l;
            logit_max = std::max(logit_max, l);
            logit_min = std::min(logit_min, l);
        }
        
        float logit_mean = logit_sum / logits.size();
        float logit_var = logit_sq_sum / logits.size() - logit_mean * logit_mean;
        float logit_std = std::sqrt(std::max(0.0f, logit_var));
        
        result["logit_mean"] = logit_mean;
        result["logit_std"] = logit_std;
        result["logit_max"] = logit_max;
        result["logit_min"] = logit_min;
        result["logit_range"] = logit_max - logit_min;
        
        // Get probability distribution statistics
        std::vector<float> probs = gpu_decoder_->decode(broca_output);
        
        float prob_max = -1e30f;
        float entropy = 0.0f;
        int argmax = 0;
        
        for (size_t i = 0; i < probs.size(); ++i) {
            if (probs[i] > prob_max) {
                prob_max = probs[i];
                argmax = i;
            }
            if (probs[i] > 1e-10f) {
                entropy -= probs[i] * std::log(probs[i]);
            }
        }
        
        result["prob_max"] = prob_max;
        result["prob_argmax"] = argmax;
        result["entropy"] = entropy;
        result["entropy_bits"] = entropy / std::log(2.0f);
        result["max_entropy_bits"] = std::log2(static_cast<float>(vocab_size_));
        
        return result;
    }

    /**
     * Set decoder learning rate
     */
    void set_decoder_learning_rate(float lr) {
        gpu_decoder_->setLearningRate(lr);
        std::cout << "📊 Decoder learning rate set to: " << lr << std::endl;
    }

    /**
     * Pre-train decoder projection matrix with random Broca-like patterns
     * This teaches the decoder to map sparse spike patterns to vocabulary
     * 
     * @param num_iterations Number of training iterations
     * @param learning_rate Learning rate for SGD
     * @return Average loss over training
     */
    float pretrain_decoder(int num_iterations, float learning_rate) {
        std::cout << "\n🎯 Pre-training decoder projection matrix..." << std::endl;
        std::cout << "   Iterations: " << num_iterations << std::endl;
        std::cout << "   Learning rate: " << learning_rate << std::endl;
        
        // Set learning rate
        gpu_decoder_->setLearningRate(learning_rate);
        
        float total_loss = 0.0f;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> token_dist(0, vocab_size_ - 1);
        std::uniform_real_distribution<float> spike_dist(0.0f, 1.0f);
        
        const int broca_dim = 8192;  // Broca output size
        const float sparsity = 0.1f; // ~10% neurons active (biologically realistic)
        
        for (int iter = 0; iter < num_iterations; iter++) {
            // Generate random target token
            int target_token = token_dist(gen);
            
            // Generate random sparse Broca-like pattern
            std::vector<float> broca_pattern(broca_dim, 0.0f);
            for (int i = 0; i < broca_dim; i++) {
                if (spike_dist(gen) < sparsity) {
                    broca_pattern[i] = 1.0f; // Spike
                }
            }
            
            // Forward pass: get probabilities
            std::vector<float> probs = gpu_decoder_->decode(broca_pattern);
            
            // Compute cross-entropy loss
            const float eps = 1e-8f;
            float loss = -std::log(std::max(probs[target_token], eps));
            total_loss += loss;
            
            // Backward pass: update projection matrix
            gpu_decoder_->trainStep(broca_pattern, target_token, probs);
            
            // Progress logging
            if ((iter + 1) % 500 == 0 || iter == 0) {
                float avg_loss = total_loss / (iter + 1);
                std::cout << "   Iter " << (iter + 1) << "/" << num_iterations 
                          << " - Loss: " << loss << " (avg: " << avg_loss << ")" << std::endl;
            }
        }
        
        float avg_loss = total_loss / num_iterations;
        std::cout << "✅ Decoder pre-training complete - Final avg loss: " << avg_loss << std::endl;
        
        return avg_loss;
    }

    /**
     * Pre-train token embeddings using Skip-gram style co-occurrence learning
     * This teaches the encoder to group similar tokens together
     * 
     * @param token_sequences List of token ID sequences from the corpus
     * @param window_size Context window size for co-occurrence
     * @param learning_rate Learning rate for SGD
     * @param negative_samples Number of negative samples per positive pair
     * @return Average loss over training
     */
    float pretrain_embeddings(const std::vector<std::vector<int>>& token_sequences, 
                               int window_size, 
                               float learning_rate,
                               int negative_samples) {
        std::cout << "\n📚 Pre-training token embeddings..." << std::endl;
        std::cout << "   Sequences: " << token_sequences.size() << std::endl;
        std::cout << "   Window size: " << window_size << std::endl;
        std::cout << "   Learning rate: " << learning_rate << std::endl;
        std::cout << "   Negative samples: " << negative_samples << std::endl;
        
        if (token_sequences.empty()) {
            std::cout << "⚠️  No sequences provided for embedding pre-training" << std::endl;
            return 0.0f;
        }
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> token_dist(0, vocab_size_ - 1);
        
        float total_loss = 0.0f;
        int num_updates = 0;
        
        // Process each sequence
        for (size_t seq_idx = 0; seq_idx < token_sequences.size(); seq_idx++) {
            const auto& sequence = token_sequences[seq_idx];
            
            // For each token in the sequence
            for (size_t i = 0; i < sequence.size(); i++) {
                int center_token = sequence[i];
                if (center_token < 0 || center_token >= vocab_size_) continue;
                
                // Get center embedding
                std::vector<float> center_emb = embedding_->getEmbedding(center_token);
                
                // Process context window
                for (int offset = -window_size; offset <= window_size; offset++) {
                    if (offset == 0) continue;
                    
                    int context_pos = static_cast<int>(i) + offset;
                    if (context_pos < 0 || context_pos >= static_cast<int>(sequence.size())) {
                        continue;
                    }
                    
                    int context_token = sequence[context_pos];
                    if (context_token < 0 || context_token >= vocab_size_) continue;
                    
                    // Get context embedding
                    std::vector<float> context_emb = embedding_->getEmbedding(context_token);
                    
                    // === Positive pair: center and context should be similar ===
                    // Compute dot product
                    float dot = 0.0f;
                    for (int d = 0; d < embedding_dim_; d++) {
                        dot += center_emb[d] * context_emb[d];
                    }
                    
                    // Sigmoid and loss
                    float sigmoid_pos = 1.0f / (1.0f + std::exp(-dot));
                    float loss_pos = -std::log(std::max(sigmoid_pos, 1e-8f));
                    
                    // Gradient for positive: (sigmoid - 1) * context_emb
                    float grad_scale_pos = (sigmoid_pos - 1.0f) * learning_rate;
                    
                    // Update center embedding
                    for (int d = 0; d < embedding_dim_; d++) {
                        center_emb[d] -= grad_scale_pos * context_emb[d];
                    }
                    
                    total_loss += loss_pos;
                    
                    // === Negative samples: center and random tokens should be dissimilar ===
                    for (int neg = 0; neg < negative_samples; neg++) {
                        int neg_token = token_dist(gen);
                        if (neg_token == center_token || neg_token == context_token) continue;
                        
                        std::vector<float> neg_emb = embedding_->getEmbedding(neg_token);
                        
                        // Compute dot product
                        float dot_neg = 0.0f;
                        for (int d = 0; d < embedding_dim_; d++) {
                            dot_neg += center_emb[d] * neg_emb[d];
                        }
                        
                        // Sigmoid (we want this to be 0)
                        float sigmoid_neg = 1.0f / (1.0f + std::exp(-dot_neg));
                        float loss_neg = -std::log(std::max(1.0f - sigmoid_neg, 1e-8f));
                        
                        // Gradient for negative: sigmoid * neg_emb
                        float grad_scale_neg = sigmoid_neg * learning_rate;
                        
                        // Update center embedding (push away from negative)
                        for (int d = 0; d < embedding_dim_; d++) {
                            center_emb[d] -= grad_scale_neg * neg_emb[d];
                        }
                        
                        total_loss += loss_neg;
                    }
                    
                    num_updates++;
                }
                
                // Normalize and save updated center embedding
                center_emb = TokenEmbedding::normalize(center_emb);
                embedding_->updateEmbedding(center_token, center_emb);
            }
            
            // Progress logging
            if ((seq_idx + 1) % 100 == 0 || seq_idx == 0) {
                float avg_loss = num_updates > 0 ? total_loss / num_updates : 0.0f;
                std::cout << "   Sequence " << (seq_idx + 1) << "/" << token_sequences.size()
                          << " - Avg loss: " << avg_loss << std::endl;
            }
        }
        
        float avg_loss = num_updates > 0 ? total_loss / num_updates : 0.0f;
        std::cout << "✅ Embedding pre-training complete - Final avg loss: " << avg_loss << std::endl;
        std::cout << "   Total updates: " << num_updates << std::endl;
        
        return avg_loss;
    }

    /**
     * Run sensory pathway pre-training
     * Train embeddings -> Thalamus -> Wernicke pathway to produce meaningful activations
     * 
     * @param token_sequences Token sequences from corpus
     * @param num_epochs Number of training epochs
     * @param learning_rate Learning rate
     * @return Average loss
     */
    float pretrain_sensory_pathway(const std::vector<std::vector<int>>& token_sequences,
                                    int num_epochs,
                                    float learning_rate) {
        // Suppress unused parameter warning
        (void)learning_rate;

        std::cout << "\n🧠 Pre-training sensory pathway (Embeddings -> Thalamus -> Wernicke)..." << std::endl;
        std::cout << "   Sequences: " << token_sequences.size() << std::endl;
        std::cout << "   Epochs: " << num_epochs << std::endl;
        
        float total_loss = 0.0f;
        int num_steps = 0;
        
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            float epoch_loss = 0.0f;
            int epoch_steps = 0;
            
            for (const auto& sequence : token_sequences) {
                // Reset brain state for each sequence
                brain_->reset();
                
                for (size_t i = 0; i + 1 < sequence.size(); i++) {
                    int input_token = sequence[i];
                    int target_token = sequence[i + 1];
                    
                    if (input_token < 0 || input_token >= vocab_size_) continue;
                    if (target_token < 0 || target_token >= vocab_size_) continue;
                    
                    // Get input embedding
                    std::vector<float> input_emb = embedding_->encodeById(input_token);
                    
                    // Forward pass through brain
                    brain_->cognitiveStep(input_emb);
                    
                    // Get Broca output and decode
                    std::vector<float> broca_output = brain_->getBrocaOutput();
                    std::vector<float> probs = gpu_decoder_->decode(broca_output);
                    
                    // Compute loss
                    const float eps = 1e-8f;
                    float loss = -std::log(std::max(probs[target_token], eps));
                    epoch_loss += loss;
                    total_loss += loss;
                    
                    // Train decoder
                    gpu_decoder_->trainStep(broca_output, target_token, probs);
                    
                    // Apply small reward based on probability of correct token
                    float reward = probs[target_token] - (1.0f / vocab_size_);
                    brain_->distributeReward(reward * 0.1f);
                    
                    epoch_steps++;
                    num_steps++;
                }
            }
            
            float avg_epoch_loss = epoch_steps > 0 ? epoch_loss / epoch_steps : 0.0f;
            std::cout << "   Epoch " << (epoch + 1) << "/" << num_epochs 
                      << " - Avg loss: " << avg_epoch_loss << std::endl;
        }
        
        float avg_loss = num_steps > 0 ? total_loss / num_steps : 0.0f;
        std::cout << "✅ Sensory pathway pre-training complete - Final avg loss: " << avg_loss << std::endl;
        
        return avg_loss;
    }

private:
    int vocab_size_;
    int embedding_dim_;
    int gpu_device_;
    std::unique_ptr<BrainOrchestrator> brain_;
    std::shared_ptr<TokenEmbedding> embedding_;
    std::shared_ptr<GPUDecoder> gpu_decoder_;        // Training decoder (GREEDY)
    std::shared_ptr<GPUDecoder> inference_decoder_;  // Inference decoder (TOP_K)
};

// Python module definition
PYBIND11_MODULE(libneurogen, m) {
    m.doc() = "NeuroGen 2.0 - Bio-inspired Neural Language Model";
    
    py::class_<NeuroGenModel>(m, "NeuroGenModel")
        .def(py::init<int, int, int>(),
             py::arg("vocab_size"),
             py::arg("embedding_dim"),
             py::arg("gpu_device") = 0,
             "Initialize NeuroGen model\n\n"
             "Args:\n"
             "    vocab_size: Size of vocabulary\n"
             "    embedding_dim: Dimension of token embeddings\n"
             "    gpu_device: GPU device ID (default: 0)")
        .def("process_input", [](NeuroGenModel& self, char input_char) {
            // Convert char to embedding
            int token_id = static_cast<int>(static_cast<unsigned char>(input_char));
            std::vector<float> embedded = self.getEmbedding()->encodeById(token_id);
            
            // Run cognitive step
            return self.getBrain()->cognitiveStep(embedded);
        }, "Process a single character input")
        
        .def("predict_next", [](NeuroGenModel& self) {
            // Get output from Broca's area
            std::vector<float> brain_output = self.getBrain()->getBrocaOutput();
            
            // Decode to probabilities
            auto token_and_prob = self.getGPUDecoder()->decodeAndSampleWithProb(brain_output);
            
            // Convert token ID back to char
            char predicted_char = static_cast<char>(token_and_prob.first);
            float confidence = token_and_prob.second;
            
            return std::make_pair(predicted_char, confidence);
        }, "Predict the next character based on current state")
        
        .def("apply_reward", [](NeuroGenModel& self, float reward) {
            self.getBrain()->distributeReward(reward);
            // Modulate global state based on reward (dopamine/serotonin)
            // Positive reward -> Dopamine (learning/action)
            // Negative reward -> Serotonin (inhibition/caution)
            float dopamine = std::max(0.0f, reward);
            float serotonin = std::max(0.0f, -reward);
            self.getBrain()->modulateGlobalState(dopamine, serotonin, 0.0f);
        }, "Apply reward signal to the brain")
        
        .def("set_plasticity_mode", [](NeuroGenModel& self, const std::string& mode) {
            // Suppress unused parameter warning
            (void)self;
            // This would interface with CorticalModule config to change learning rules
            // For now, we just print it, but in a full implementation this would
            // adjust STDP parameters in the C++ engine
            std::cout << "🧠 Switching plasticity mode to: " << mode << std::endl;
            // TODO: Implement actual parameter switching in BrainOrchestrator
        }, "Set the developmental plasticity mode")
        
        .def("train_step", &NeuroGenModel::train_step,
             py::arg("input_ids"),
             py::arg("target_ids"),
             "Train on a sequence with recurrent processing\n\n"
             "The model's internal working memory (PFC, Hippocampus, pipeline state)\n"
             "maintains temporal context automatically between calls.\n\n"
             "Args:\n"
             "    input_ids: List of input token IDs (full context)\n"
             "    target_ids: List of target token IDs (we use last one)\n\n"
             "Returns:\n"
             "    Tuple of (loss, accuracy, predicted_token_id)")
        .def("generate", &NeuroGenModel::generate,
             py::arg("prompt_ids"),
             py::arg("max_length") = 100,
             "Generate text from a prompt\n\n"
             "Args:\n"
             "    prompt_ids: List of prompt token IDs\n"
             "    max_length: Maximum generation length\n\n"
             "Returns:\n"
             "    List of generated token IDs")
        .def("save_checkpoint", &NeuroGenModel::save_checkpoint,
             py::arg("path"),
             "Save model checkpoint")
        .def("load_checkpoint", &NeuroGenModel::load_checkpoint,
             py::arg("path"),
             "Load model checkpoint")
        .def("get_statistics", &NeuroGenModel::get_statistics,
             "Get brain statistics\n\n"
             "Returns:\n"
             "    Dictionary of brain statistics")
        .def("get_diagnostics", &NeuroGenModel::get_diagnostics,
             "Get diagnostic information about neural activity and decoder\n\n"
             "Returns:\n"
             "    Dictionary with Broca output stats, logit stats, and probability stats")
        .def("set_decoder_learning_rate", &NeuroGenModel::set_decoder_learning_rate,
             py::arg("lr"),
             "Set the decoder learning rate\n\n"
             "Args:\n"
             "    lr: Learning rate (default: 0.001)")
        .def("pretrain_decoder", &NeuroGenModel::pretrain_decoder,
             py::arg("num_iterations") = 5000,
             py::arg("learning_rate") = 0.01f,
             "Pre-train decoder projection matrix to map Broca patterns to vocabulary\n\n"
             "This teaches the decoder to distinguish between different sparse spike patterns\n"
             "and map them to different vocabulary tokens.\n\n"
             "Args:\n"
             "    num_iterations: Number of training iterations (default: 5000)\n"
             "    learning_rate: Learning rate for SGD (default: 0.01)\n\n"
             "Returns:\n"
             "    Average loss over training")
        .def("pretrain_embeddings", &NeuroGenModel::pretrain_embeddings,
             py::arg("token_sequences"),
             py::arg("window_size") = 5,
             py::arg("learning_rate") = 0.01f,
             py::arg("negative_samples") = 5,
             "Pre-train token embeddings using Skip-gram style co-occurrence learning\n\n"
             "This teaches the encoder to group semantically similar tokens together,\n"
             "making it easier for the SNN to learn meaningful patterns.\n\n"
             "Args:\n"
             "    token_sequences: List of token ID sequences from the corpus\n"
             "    window_size: Context window size (default: 5)\n"
             "    learning_rate: Learning rate for SGD (default: 0.01)\n"
             "    negative_samples: Number of negative samples per positive pair (default: 5)\n\n"
             "Returns:\n"
             "    Average loss over training")
        .def("pretrain_sensory_pathway", &NeuroGenModel::pretrain_sensory_pathway,
             py::arg("token_sequences"),
             py::arg("num_epochs") = 1,
             py::arg("learning_rate") = 0.001f,
             "Pre-train sensory pathway (Embeddings -> Thalamus -> Wernicke -> Broca -> Decoder)\n\n"
             "This trains the full pipeline end-to-end with next-token prediction,\n"
             "establishing meaningful signal flow through the neural network.\n\n"
             "Args:\n"
             "    token_sequences: List of token ID sequences from the corpus\n"
             "    num_epochs: Number of training epochs (default: 1)\n"
             "    learning_rate: Learning rate (default: 0.001)\n\n"
             "Returns:\n"
             "    Average loss over training")
        .def("train_decoder_step", [](NeuroGenModel& self, int target_token) {
            // Get current brain state
            std::vector<float> brain_output = self.getBrain()->getBrocaOutput();
            
            // Get current predictions (needed for gradient calculation)
            std::vector<float> probs = self.getGPUDecoder()->decode(brain_output);
            
            // Perform gradient descent step on decoder
            float loss = self.getGPUDecoder()->trainStep(brain_output, target_token, probs);
            
            return loss;
        }, "Train the decoder projection matrix on the current state")
        .def("get_brain", &NeuroGenModel::getBrain, py::return_value_policy::reference)
        .def("get_embedding", &NeuroGenModel::getEmbeddingPtr, py::return_value_policy::reference)
        .def("get_gpu_decoder", &NeuroGenModel::getGPUDecoderPtr, py::return_value_policy::reference);
    
    // Version info
    m.attr("__version__") = "2.0.0";
}

