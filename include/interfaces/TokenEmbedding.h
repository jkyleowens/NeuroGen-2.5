#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

/**
 * @brief Token embedding interface for converting text to neural input vectors
 * 
 * This class provides vocabulary management and embedding lookups for converting
 * text tokens into dense vector representations suitable for neural processing.
 */
class TokenEmbedding {
public:
    struct Config {
        int vocab_size;          // Size of vocabulary
        int embedding_dim;       // Dimensionality of embeddings (512)
        bool use_random_init;    // Random initialization vs pretrained
        float normalization;     // L2 normalization factor
        std::string vocab_file;  // Path to vocabulary file (optional)
    };

    TokenEmbedding(const Config& config);
    ~TokenEmbedding() = default;

    /**
     * @brief Initialize embeddings (random or from file)
     */
    void initialize();

    /**
     * @brief Encode a single token to embedding vector
     * @param token String token to encode
     * @return Embedding vector
     */
    std::vector<float> encode(const std::string& token);

    /**
     * @brief Encode token by ID
     * @param token_id Token ID in vocabulary
     * @return Embedding vector
     */
    std::vector<float> encodeById(int token_id);

    /**
     * @brief Encode a sequence of tokens
     * @param tokens Vector of string tokens
     * @return Vector of embedding vectors
     */
    std::vector<std::vector<float>> encodeSequence(const std::vector<std::string>& tokens);

    /**
     * @brief Get token ID from string
     * @param token String token
     * @return Token ID or -1 if not found
     */
    int getTokenId(const std::string& token) const;

    /**
     * @brief Get token string from ID
     * @param token_id Token ID
     * @return Token string or "<UNK>" if not found
     */
    std::string getToken(int token_id) const;

    /**
     * @brief Add token to vocabulary
     * @param token New token to add
     * @return Token ID assigned
     */
    int addToken(const std::string& token);

    /**
     * @brief Load vocabulary from file
     * @param filepath Path to vocabulary file
     */
    void loadVocabulary(const std::string& filepath);

    /**
     * @brief Save vocabulary to file
     * @param filepath Path to save vocabulary
     */
    void saveVocabulary(const std::string& filepath);

    /**
     * @brief Load pretrained embeddings
     * @param filepath Path to embeddings file
     */
    void loadEmbeddings(const std::string& filepath);

    /**
     * @brief Save embeddings to file
     * @param filepath Path to save embeddings
     */
    void saveEmbeddings(const std::string& filepath);

    /**
     * @brief Get embedding dimension
     */
    int getEmbeddingDim() const { return config_.embedding_dim; }

    /**
     * @brief Get vocabulary size
     */
    int getVocabSize() const { return config_.vocab_size; }

    /**
     * @brief Update a single token's embedding (for pre-training)
     * @param token_id Token ID to update
     * @param new_embedding New embedding vector
     */
    void updateEmbedding(int token_id, const std::vector<float>& new_embedding);

    /**
     * @brief Get a copy of a single embedding vector
     * @param token_id Token ID
     * @return Embedding vector (unnormalized)
     */
    std::vector<float> getEmbedding(int token_id) const;

    /**
     * @brief Normalize embedding vector
     */
    static std::vector<float> normalize(const std::vector<float>& embedding);

private:
    Config config_;
    
    // Vocabulary mappings
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<int, std::string> id_to_token_;
    
    // Embedding matrix: [vocab_size x embedding_dim]
    std::vector<std::vector<float>> embeddings_;
    
    int next_token_id_;
    
    // Special tokens
    static constexpr const char* UNK_TOKEN = "<UNK>";
    static constexpr const char* PAD_TOKEN = "<PAD>";
    static constexpr const char* BOS_TOKEN = "<BOS>";
    static constexpr const char* EOS_TOKEN = "<EOS>";
    
    // Initialize special tokens
    void initializeSpecialTokens();
    
    // Random initialization helper
    std::vector<float> randomEmbedding();
};

