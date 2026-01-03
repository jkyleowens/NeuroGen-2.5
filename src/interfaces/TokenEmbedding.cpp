#include "interfaces/TokenEmbedding.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <algorithm>

TokenEmbedding::TokenEmbedding(const Config& config)
    : config_(config), next_token_id_(0) {
    
    embeddings_.reserve(config_.vocab_size);
    
    std::cout << "📝 Initializing TokenEmbedding with vocab_size=" 
              << config_.vocab_size << ", embedding_dim=" 
              << config_.embedding_dim << std::endl;
}

void TokenEmbedding::initialize() {
    // Clear existing embeddings
    embeddings_.clear();
    embeddings_.resize(256); // Fixed size for ASCII/Byte values

    std::cout << "🔤 Initializing Direct Character Embeddings (Fixed One-Hot)" << std::endl;

    // Create fixed one-hot embeddings for each character
    for (int i = 0; i < 256; ++i) {
        std::vector<float> embedding(config_.embedding_dim, 0.0f);
        
        // Direct mapping: Input channel i corresponds to character i
        // We use a strong signal value (e.g., 1.0 or higher)
        if (i < config_.embedding_dim) {
            embedding[i] = 1.0f;
        }
        
        embeddings_[i] = embedding;
    }
    
    std::cout << "✓ Initialized 256 direct character embeddings" << std::endl;
}

void TokenEmbedding::initializeSpecialTokens() {
    // No special tokens in direct character mode - every byte is valid
}

std::vector<float> TokenEmbedding::encode(const std::string& token) {
    if (token.empty()) {
        return std::vector<float>(config_.embedding_dim, 0.0f);
    }
    
    // Use the first character of the token
    unsigned char char_code = static_cast<unsigned char>(token[0]);
    return encodeById(static_cast<int>(char_code));
}

std::vector<float> TokenEmbedding::encodeById(int token_id) {
    if (token_id < 0 || token_id >= 256) {
        return std::vector<float>(config_.embedding_dim, 0.0f);
    }
    
    return embeddings_[token_id];
}

std::vector<std::vector<float>> TokenEmbedding::encodeSequence(
    const std::vector<std::string>& tokens) {
    
    std::vector<std::vector<float>> sequence_embeddings;
    sequence_embeddings.reserve(tokens.size());
    
    for (const auto& token : tokens) {
        sequence_embeddings.push_back(encode(token));
    }
    
    return sequence_embeddings;
}

int TokenEmbedding::getTokenId(const std::string& token) const {
    if (token.empty()) return 0;
    return static_cast<int>(static_cast<unsigned char>(token[0]));
}

std::string TokenEmbedding::getToken(int token_id) const {
    if (token_id < 0 || token_id >= 256) return "";
    return std::string(1, static_cast<char>(token_id));
}

int TokenEmbedding::addToken(const std::string& token) {
    // Not applicable in direct character mode
    return -1;
}

void TokenEmbedding::loadVocabulary(const std::string& filepath) {
    // Not applicable in direct character mode
}

void TokenEmbedding::saveVocabulary(const std::string& filepath) {
    // Not applicable in direct character mode
}

void TokenEmbedding::loadEmbeddings(const std::string& filepath) {
    // Not applicable in direct character mode
}

void TokenEmbedding::saveEmbeddings(const std::string& filepath) {
    // Not applicable in direct character mode
}

std::vector<float> TokenEmbedding::normalize(const std::vector<float>& embedding) {
    // L2 normalization
    float sum_sq = 0.0f;
    for (float val : embedding) {
        sum_sq += val * val;
    }
    
    float norm = std::sqrt(sum_sq);
    if (norm < 1e-6f) {
        return embedding;  // Avoid division by zero
    }
    
    std::vector<float> normalized(embedding.size());
    for (size_t i = 0; i < embedding.size(); ++i) {
        normalized[i] = embedding[i] / norm;
    }
    
    return normalized;
}

void TokenEmbedding::updateEmbedding(int token_id, const std::vector<float>& new_embedding) {
    // Not applicable in direct character mode
}

std::vector<float> TokenEmbedding::getEmbedding(int token_id) const {
    if (token_id < 0 || token_id >= 256) {
        return std::vector<float>(config_.embedding_dim, 0.0f);
    }
    return embeddings_[token_id];
}

std::vector<float> TokenEmbedding::randomEmbedding() {
    // Not applicable in direct character mode
    return std::vector<float>(config_.embedding_dim, 0.0f);
}

