#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "modules/BrainOrchestrator.h"
#include "interfaces/TokenEmbedding.h"
#include "interfaces/OutputDecoder.h"

namespace py = pybind11;

/**
 * @brief Python wrapper for the NeuroGen Modular Brain
 * 
 * This binding exposes the core functionality needed for training:
 * - Token embedding
 * - Model forward pass
 * - Training step with next-token prediction
 * - State management
 */
class NeuroGenModel {
public:
    NeuroGenModel(int vocab_size = 10000, int embedding_dim = 512, int gpu_device = 0) {
        // Initialize orchestrator
        BrainOrchestrator::Config brain_config;
        brain_config.time_step_ms = 1.0f;
        brain_config.enable_consolidation = true;
        brain_config.consolidation_interval_ms = 10000.0f;
        brain_config.gpu_device_id = gpu_device;

        orchestrator_ = std::make_shared<BrainOrchestrator>(brain_config);
        orchestrator_->initializeModules();
        orchestrator_->createConnectome();

        // Initialize token embedding
        TokenEmbedding::Config embedding_config;
        embedding_config.vocab_size = vocab_size;
        embedding_config.embedding_dim = embedding_dim;
        embedding_config.use_random_init = true;
        embedding_config.normalization = 1.0f;
        embedding_config.vocab_file.clear();

        token_embedding_ = std::make_shared<TokenEmbedding>(embedding_config);
        token_embedding_->initialize();

        // Initialize output decoder
        OutputDecoder::Config decoder_config;
        decoder_config.vocab_size = vocab_size;
        decoder_config.output_dim = embedding_dim;
        decoder_config.temperature = 1.0f;
        decoder_config.top_k = 50;
        decoder_config.top_p = 0.95f;
        decoder_config.beam_width = 5;
        decoder_config.strategy = OutputDecoder::SamplingStrategy::TEMPERATURE;

        output_decoder_ = std::make_shared<OutputDecoder>(decoder_config, token_embedding_);

        vocab_size_ = vocab_size;
        embedding_dim_ = embedding_dim;
    }
    
    /**
     * @brief Encode tokens to embeddings
     * @param token_ids List of token IDs
     * @return Flattened embedding vector
     */
    std::vector<float> encode_tokens(const std::vector<int>& token_ids) {
        std::vector<float> packed;
        packed.reserve(token_ids.size() * embedding_dim_);

        for (int token_id : token_ids) {
            auto embedding = token_embedding_->encodeById(token_id);
            packed.insert(packed.end(), embedding.begin(), embedding.end());
        }

        return packed;
    }
    
    /**
     * @brief Forward pass through the brain
     * @param embedding Input embedding vector
     * @return Output probabilities (vocab_size)
     */
    std::vector<float> forward(const std::vector<float>& embedding) {
        // Perform cognitive step
        auto brain_output = orchestrator_->cognitiveStep(embedding);
        
        // Decode to probabilities
        return output_decoder_->decode(brain_output);
    }
    
    /**
     * @brief Training step with next-token prediction
     * @param input_token_ids Input sequence token IDs
     * @param target_token_ids Target sequence token IDs (shifted by 1)
     * @return Pair of (loss, accuracy)
     */
    std::pair<float, float> train_step(const std::vector<int>& input_token_ids, 
                     const std::vector<int>& target_token_ids) {
        
        if (input_token_ids.empty() || target_token_ids.empty()) {
            return {0.0f, 0.0f};
        }
        
        float total_loss = 0.0f;
        int correct_predictions = 0;
        size_t num_predictions = std::min(input_token_ids.size(), target_token_ids.size());
        
        for (size_t i = 0; i < num_predictions; ++i) {
            // Encode input token
            auto embedding = token_embedding_->encodeById(input_token_ids[i]);
            
            // Forward pass
            auto brain_output = orchestrator_->cognitiveStep(embedding);
            auto probabilities = output_decoder_->decode(brain_output);
            
            // Compute cross-entropy loss
            int target_token = target_token_ids[i];
            float loss = compute_cross_entropy_loss(probabilities, target_token);
            total_loss += loss;
            
            // Check accuracy (argmax)
            int predicted_token = 0;
            float max_prob = -1.0f;
            for (size_t j = 0; j < probabilities.size(); ++j) {
                if (probabilities[j] > max_prob) {
                    max_prob = probabilities[j];
                    predicted_token = static_cast<int>(j);
                }
            }
            
            if (predicted_token == target_token) {
                correct_predictions++;
            }
            
            // Compute reward signal (negative loss)
            float reward = -loss;
            
            // Distribute reward for learning
            orchestrator_->distributeReward(reward);
        }
        
        float avg_loss = total_loss / num_predictions;
        float accuracy = static_cast<float>(correct_predictions) / num_predictions;
        
        return {avg_loss, accuracy};
    }
    
    /**
     * @brief Sample next token from logits
     * @param logits Output logits
     * @param temperature Sampling temperature
     * @return Sampled token ID
     */
    int sample_token(const std::vector<float>& probabilities, float temperature = 1.0f) {
        float previous_temp = output_decoder_->getConfig().temperature;
        output_decoder_->setTemperature(temperature);
        int token = output_decoder_->sampleToken(probabilities);
        output_decoder_->setTemperature(previous_temp);
        return token;
    }
    
    /**
     * @brief Generate text from prompt
     * @param prompt_token_ids Prompt token IDs
     * @param max_length Maximum generation length
     * @param temperature Sampling temperature
     * @return Generated token IDs
     */
    std::vector<int> generate(const std::vector<int>& prompt_token_ids,
                             int max_length = 100,
                             float temperature = 1.0f) {
        std::vector<int> generated = prompt_token_ids;
        
        for (int i = 0; i < max_length; ++i) {
            // Get last token
            int current_token = generated.back();
            
            // Encode and forward
            auto embedding = token_embedding_->encodeById(current_token);
            auto brain_output = orchestrator_->cognitiveStep(embedding);
            auto probabilities = output_decoder_->decode(brain_output);
            
            // Sample next token
            int next_token = sample_token(probabilities, temperature);
            
            // Stop at end token (assuming 0 is end token)
            if (next_token == 0) {
                break;
            }
            
            generated.push_back(next_token);
        }
        
        return generated;
    }
    
    /**
     * @brief Save model checkpoint
     * @param checkpoint_path Path to save checkpoint
     */
    void save_checkpoint(const std::string& checkpoint_path) {
        std::ofstream ofs(checkpoint_path);
        if (!ofs) {
            throw std::runtime_error("Unable to open checkpoint path: " + checkpoint_path);
        }

        auto stats = orchestrator_->getStats();
        ofs << "{\n";
        ofs << "  \"tokens_processed\": " << stats.tokens_processed << ",\n";
        ofs << "  \"cognitive_cycles\": " << stats.cognitive_cycles << ",\n";
        ofs << "  \"average_reward\": " << stats.average_reward << ",\n";
        ofs << "  \"total_time_ms\": " << stats.total_time_ms << "\n";
        ofs << "}\n";
    }
    
    /**
     * @brief Load model checkpoint
     * @param checkpoint_path Path to load checkpoint
     */
    void load_checkpoint(const std::string& checkpoint_path) {
        std::ifstream ifs(checkpoint_path);
        if (!ifs) {
            throw std::runtime_error("Unable to open checkpoint path: " + checkpoint_path);
        }
        // Currently a stub – consuming file to validate existence.
        std::string line;
        while (std::getline(ifs, line)) {
            (void)line;
        }
    }
    
    /**
     * @brief Get training statistics
     * @return Dictionary of statistics
     */
    py::dict get_statistics() {
        py::dict stat_map;
        auto stats = orchestrator_->getStats();

        stat_map["total_time_ms"] = stats.total_time_ms;
        stat_map["cognitive_cycles"] = static_cast<float>(stats.cognitive_cycles);
        stat_map["tokens_processed"] = static_cast<float>(stats.tokens_processed);
        stat_map["average_reward"] = stats.average_reward;
        stat_map["current_phase"] = static_cast<float>(stats.current_phase);
        stat_map["vocab_size"] = static_cast<float>(vocab_size_);
        stat_map["embedding_dim"] = static_cast<float>(embedding_dim_);

        py::dict modules_dict;
        for (const auto& [module_name, mod_stats] : stats.module_stats) {
            py::dict mod_info;
            mod_info["activity"] = mod_stats.activity_level;
            mod_info["dopamine"] = mod_stats.dopamine_level;
            mod_info["serotonin"] = mod_stats.serotonin_level;
            modules_dict[module_name.c_str()] = mod_info;
        }
        stat_map["modules"] = modules_dict;

        return stat_map;
    }
    
    /**
     * @brief Reset model state
     */
    void reset() {
        orchestrator_->reset();
    }
    
private:
    /**
     * @brief Compute cross-entropy loss
     * @param logits Output logits
     * @param target_token Target token ID
     * @return Loss value
     */
    float compute_cross_entropy_loss(const std::vector<float>& probabilities, int target_token) {
        if (target_token < 0 || target_token >= static_cast<int>(probabilities.size())) {
            return 0.0f;
        }

        float prob = std::max(probabilities[target_token], 1e-8f);
        return -std::log(prob);
    }
    
    std::shared_ptr<BrainOrchestrator> orchestrator_;
    std::shared_ptr<TokenEmbedding> token_embedding_;
    std::shared_ptr<OutputDecoder> output_decoder_;
    
    int vocab_size_;
    int embedding_dim_;
};


PYBIND11_MODULE(libneurogen, m) {
    m.doc() = "NeuroGen Modular Brain - Python Binding for Training";
    
    py::class_<NeuroGenModel>(m, "NeuroGenModel")
        .def(py::init<int, int, int>(),
             py::arg("vocab_size") = 10000,
             py::arg("embedding_dim") = 512,
             py::arg("gpu_device") = 0,
             "Initialize NeuroGen model")
        
        .def("encode_tokens", &NeuroGenModel::encode_tokens,
             py::arg("token_ids"),
             "Encode token IDs to embeddings")
        
        .def("forward", &NeuroGenModel::forward,
             py::arg("embedding"),
             "Forward pass through the brain")
        
        .def("train_step", &NeuroGenModel::train_step,
             py::arg("input_token_ids"),
             py::arg("target_token_ids"),
             "Training step with next-token prediction")
        
        .def("sample_token", &NeuroGenModel::sample_token,
             py::arg("logits"),
             py::arg("temperature") = 1.0f,
             "Sample next token from logits")
        
        .def("generate", &NeuroGenModel::generate,
             py::arg("prompt_token_ids"),
             py::arg("max_length") = 100,
             py::arg("temperature") = 1.0f,
             "Generate text from prompt")
        
        .def("save_checkpoint", &NeuroGenModel::save_checkpoint,
             py::arg("checkpoint_path"),
             "Save model checkpoint")
        
        .def("load_checkpoint", &NeuroGenModel::load_checkpoint,
             py::arg("checkpoint_path"),
             "Load model checkpoint")
        
        .def("get_statistics", &NeuroGenModel::get_statistics,
             "Get training statistics")
        
        .def("reset", &NeuroGenModel::reset,
             "Reset model state");
}

