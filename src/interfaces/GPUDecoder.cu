#include "interfaces/GPUDecoder.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <random>
#include <fstream>
#include <vector>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t stat = call; \
        if (stat != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error("cuBLAS error: " + std::to_string(stat)); \
        } \
    } while(0)

// ============================================================================
// GPU Kernels
// ============================================================================

/**
 * @brief Add bias to logits (element-wise)
 */
__global__ void addBiasKernel(float* __restrict__ logits,
                              const float* __restrict__ bias,
                              int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vocab_size) {
        logits[idx] += bias[idx];
    }
}

/**
 * @brief Softmax kernel with numerical stability
 * Uses shared memory for reduction and prevents overflow
 */
__global__ void softmaxKernel(const float* __restrict__ logits, 
                               float* __restrict__ probs, 
                               int vocab_size) {
    extern __shared__ float shared_data[];
    float* shared_max = shared_data;
    float* shared_sum = shared_data + blockDim.x;
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    // Find max (for numerical stability)
    float local_max = -INFINITY;
    for (int i = tid; i < vocab_size; i += stride) {
        local_max = fmaxf(local_max, logits[i]);
    }
    shared_max[tid] = local_max;
    __syncthreads();
    
    // Reduce to find global max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    float max_val = shared_max[0];
    __syncthreads();
    
    // Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += stride) {
        float exp_val = expf(logits[i] - max_val);
        probs[i] = exp_val;
        local_sum += exp_val;
    }
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    // Reduce to find total sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    float sum_val = shared_sum[0];
    __syncthreads();
    
    // Normalize
    for (int i = tid; i < vocab_size; i += stride) {
        probs[i] /= sum_val;
    }
}

/**
 * @brief Temperature-scaled softmax kernel
 */
__global__ void temperatureSoftmaxKernel(const float* __restrict__ logits,
                                          float* __restrict__ probs,
                                          int vocab_size,
                                          float temperature) {
    extern __shared__ float shared_data[];
    float* shared_max = shared_data;
    float* shared_sum = shared_data + blockDim.x;
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    // Find max
    float local_max = -INFINITY;
    for (int i = tid; i < vocab_size; i += stride) {
        local_max = fmaxf(local_max, logits[i] / temperature);
    }
    shared_max[tid] = local_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    float max_val = shared_max[0];
    __syncthreads();
    
    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += stride) {
        float exp_val = expf(logits[i] / temperature - max_val);
        probs[i] = exp_val;
        local_sum += exp_val;
    }
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    float sum_val = shared_sum[0];
    __syncthreads();
    
    // Normalize
    for (int i = tid; i < vocab_size; i += stride) {
        probs[i] /= sum_val;
    }
}

/**
 * @brief Find argmax (greedy sampling)
 * Fixed to properly initialize values and handle edge cases
 */
__global__ void argmaxKernel(const float* __restrict__ probs,
                              int* result,
                              int vocab_size) {
    extern __shared__ float shared_data[];
    int* shared_idx = (int*)(shared_data + blockDim.x);
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    // Initialize with first valid value to avoid garbage
    float local_max = -INFINITY;
    int local_idx = 0;  // Default to 0
    
    // Find local max
    for (int i = tid; i < vocab_size; i += stride) {
        float val = probs[i];
        // Check for valid float (not NaN or Inf)
        if (isfinite(val) && val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }
    
    // If no valid value found, use 0
    if (!isfinite(local_max)) {
        local_max = 0.0f;
        local_idx = 0;
    }
    shared_data[tid] = local_max;
    shared_idx[tid] = local_idx;
    __syncthreads();
    
    // Reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_data[tid + s] > shared_data[tid]) {
                shared_data[tid] = shared_data[tid + s];
                shared_idx[tid] = shared_idx[tid + s];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        *result = shared_idx[0];
    }
}

/**
 * @brief Compute cumulative sum (for sampling)
 */
__global__ void cumulativeSumKernel(const float* __restrict__ probs,
                                     float* __restrict__ cumsum,
                                     int vocab_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < vocab_size) {
        float sum = 0.0f;
        for (int i = 0; i <= tid; ++i) {
            sum += probs[i];
        }
        cumsum[tid] = sum;
    }
}

/**
 * @brief Binary search for sampling
 */
__global__ void binarySearchSampleKernel(const float* __restrict__ cumsum,
                                          float random_val,
                                          int* result,
                                          int vocab_size) {
    if (threadIdx.x == 0) {
        int left = 0;
        int right = vocab_size - 1;
        
        while (left < right) {
            int mid = (left + right) / 2;
            if (cumsum[mid] < random_val) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        *result = left;
    }
}

/**
 * @brief Initialize RNG states
 */
__global__ void initRNGKernel(curandState* states, unsigned long long seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &states[tid]);
}

// ============================================================================
// GPUDecoder Implementation
// ============================================================================

GPUDecoder::GPUDecoder(const Config& config, std::shared_ptr<TokenEmbedding> embeddings)
    : config_(config),
      embeddings_(embeddings),
      gpu_device_(config.gpu_device),
      cublas_handle_(nullptr),
      stream_(nullptr),
      d_projection_matrix_(nullptr),
      d_projection_bias_(nullptr),
      d_logits_(nullptr),
      d_probabilities_(nullptr),
      d_neural_input_(nullptr),
      h_probabilities_(nullptr),
      d_rng_state_(nullptr) {
    
    initializeGPU();
    std::cout << "✓ GPU Decoder initialized (cuBLAS + CUDA)\n";
    std::cout << "  Vocab: " << config_.vocab_size << " | Output dim: " << config_.output_dim << "\n";
    std::cout << "  Expected speedup: 50-100x over CPU decoder\n";
    
    // Initialize projection matrix for direct character output
    // For direct character output, we want a 1-to-1 mapping if output_dim == vocab_size
    // Or a learned projection if they differ.
    // Since we are using direct character I/O, let's initialize it to identity-like
    // to help the model start with a reasonable mapping.
    
    std::vector<float> h_matrix(static_cast<size_t>(config_.vocab_size) * config_.output_dim, 0.0f);
    std::vector<float> h_bias(config_.vocab_size, 0.0f);
    
    // Initialize with identity-like structure if dimensions match
    // This maps neural output i directly to token i
    if (config_.vocab_size == config_.output_dim) {
        for (int i = 0; i < config_.vocab_size; ++i) {
            // Row-major: matrix[row * cols + col]
            // We want logits[i] = input[i] * 1.0
            // So W[i, i] = 1.0
            // In cuBLAS (column-major), element (i, j) is at index i + j * rows
            // We want W * x = y. W is vocab_size x output_dim.
            // cuBLAS uses column-major storage.
            // Element (row, col) is at index row + col * vocab_size
            
            // Set diagonal to 1.0 (or a scaling factor)
            h_matrix[i + i * config_.vocab_size] = 5.0f; // Strong initial signal
        }
    } else {
        // Random initialization for mismatched dimensions
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.02f);
        for (auto& val : h_matrix) val = dist(gen);
    }
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_projection_matrix_, h_matrix.data(), 
                         h_matrix.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_projection_bias_, h_bias.data(), 
                         h_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
}

GPUDecoder::~GPUDecoder() {
    cleanupGPU();
}

void GPUDecoder::initializeGPU() {
    CUDA_CHECK(cudaSetDevice(gpu_device_));
    
    // Create cuBLAS handle and stream
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream_));
    
    // Allocate GPU memory
    size_t matrix_size = static_cast<size_t>(config_.vocab_size) * config_.output_dim;
    CUDA_CHECK(cudaMalloc(&d_projection_matrix_, matrix_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_projection_bias_, config_.vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits_, config_.vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_probabilities_, config_.vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_neural_input_, config_.output_dim * sizeof(float)));
    
    // Allocate pinned host memory for fast CPU-GPU transfers
    CUDA_CHECK(cudaMallocHost(&h_probabilities_, config_.vocab_size * sizeof(float)));
    
    // Initialize RNG state
    CUDA_CHECK(cudaMalloc(&d_rng_state_, sizeof(curandState)));
    initRNGKernel<<<1, 1>>>(reinterpret_cast<curandState*>(d_rng_state_), time(nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Initialize projection matrix
    initializeProjectionMatrix();
}

void GPUDecoder::cleanupGPU() {
    if (d_projection_matrix_) cudaFree(d_projection_matrix_);
    if (d_projection_bias_) cudaFree(d_projection_bias_);
    if (d_logits_) cudaFree(d_logits_);
    if (d_probabilities_) cudaFree(d_probabilities_);
    if (d_neural_input_) cudaFree(d_neural_input_);
    if (h_probabilities_) cudaFreeHost(h_probabilities_);
    if (d_rng_state_) cudaFree(d_rng_state_);
    
    if (stream_) cudaStreamDestroy(stream_);
    if (cublas_handle_) cublasDestroy(cublas_handle_);
}

void GPUDecoder::initializeProjectionMatrix() {
    // Initialize with improved initialization for sparse binary inputs
    // 
    // Problem: Standard Xavier assumes Gaussian inputs with unit variance.
    // Our Broca output is SPARSE BINARY (0/1) with ~10% active neurons.
    // 
    // With ~10% sparsity (820 active out of 8192), the expected input sum is:
    //   E[sum] = 0.1 * 8192 = 819.2
    // 
    // To get meaningful logit variance, we need larger weights.
    // We use a scale that produces logits with std ~2-3 for good softmax spread.
    
    size_t matrix_size = static_cast<size_t>(config_.vocab_size) * config_.output_dim;
    std::vector<float> h_matrix(matrix_size);
    
    // Use a proper random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Scale for sparse binary inputs:
    // With ~10% sparsity, each logit = sum of ~820 weights
    // We want std(logit) ≈ 2-3 for good softmax discrimination
    // std(logit) = sqrt(820) * std(weight) ≈ 28.6 * std(weight)
    // So std(weight) ≈ 0.1 gives std(logit) ≈ 2.86
    float weight_std = 0.1f;
    std::normal_distribution<float> dist(0.0f, weight_std);
    
    for (size_t i = 0; i < matrix_size; ++i) {
        h_matrix[i] = dist(gen);
    }
    
    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(d_projection_matrix_, h_matrix.data(), 
                          matrix_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize bias to small values to break symmetry
    std::vector<float> h_bias(config_.vocab_size);
    std::normal_distribution<float> bias_dist(0.0f, 0.01f);
    for (int i = 0; i < config_.vocab_size; ++i) {
        h_bias[i] = bias_dist(gen);
    }
    CUDA_CHECK(cudaMemcpy(d_projection_bias_, h_bias.data(),
                          config_.vocab_size * sizeof(float), cudaMemcpyHostToDevice));
    
    std::cout << "✓ Projection matrix initialized for sparse binary inputs" << std::endl;
    std::cout << "   Weight std: " << weight_std << " (expected logit std: ~2.9)" << std::endl;
}

void GPUDecoder::projectToLogitsGPU(const float* d_input, float* d_output) {
    // Matrix-vector multiplication using cuBLAS
    // logits = W * input + b
    // W: [vocab_size, output_dim], input: [output_dim], output: [vocab_size]
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // cuBLAS GEMV: y = alpha * A * x + beta * y
    // A: [m, n], x: [n], y: [m]
    CUBLAS_CHECK(cublasSgemv(
        cublas_handle_,
        CUBLAS_OP_N,                    // No transpose
        config_.vocab_size,              // m (rows of A)
        config_.output_dim,              // n (cols of A)
        &alpha,                          // alpha
        d_projection_matrix_,            // A
        config_.vocab_size,              // lda
        d_input,                         // x
        1,                               // incx
        &beta,                           // beta
        d_output,                        // y
        1                                // incy
    ));
    
    // Add bias
    int block_size = 256;
    int num_blocks = (config_.vocab_size + block_size - 1) / block_size;
    addBiasKernel<<<num_blocks, block_size, 0, stream_>>>(
        d_output, d_projection_bias_, config_.vocab_size
    );
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void GPUDecoder::softmaxGPU(const float* d_logits, float* d_probs) {
    int block_size = 256;
    int shared_mem_size = 2 * block_size * sizeof(float);
    
    if (config_.strategy == SamplingStrategy::TEMPERATURE && config_.temperature != 1.0f) {
        temperatureSoftmaxKernel<<<1, block_size, shared_mem_size, stream_>>>(
            d_logits, d_probs, config_.vocab_size, config_.temperature
        );
    } else {
        softmaxKernel<<<1, block_size, shared_mem_size, stream_>>>(
            d_logits, d_probs, config_.vocab_size
        );
    }
    
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

int GPUDecoder::sampleTokenGPU(const float* d_probs) {
    switch (config_.strategy) {
        case SamplingStrategy::GREEDY:
            return greedySampleGPU(d_probs);
        case SamplingStrategy::TEMPERATURE:
            return temperatureSampleGPU(d_probs);
        case SamplingStrategy::TOP_K:
            return topKSampleGPU(d_probs);
        case SamplingStrategy::TOP_P:
            return topPSampleGPU(d_probs);
        default:
            return greedySampleGPU(d_probs);
    }
}

int GPUDecoder::greedySampleGPU(const float* d_probs) {
    int* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));
    
    // Initialize result to 0 to avoid garbage
    int init_val = 0;
    CUDA_CHECK(cudaMemcpy(d_result, &init_val, sizeof(int), cudaMemcpyHostToDevice));
    
    int block_size = 256;
    int shared_mem_size = block_size * (sizeof(float) + sizeof(int));
    
    argmaxKernel<<<1, block_size, shared_mem_size, stream_>>>(
        d_probs, d_result, config_.vocab_size
    );
    
    int h_result;
    CUDA_CHECK(cudaMemcpyAsync(&h_result, d_result, sizeof(int), 
                                cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    CUDA_CHECK(cudaFree(d_result));
    
    // Clamp result to valid vocab range
    if (h_result < 0 || h_result >= config_.vocab_size) {
        h_result = 0;  // Default to 0 if invalid
    }
    
    return h_result;
}

int GPUDecoder::temperatureSampleGPU(const float* d_probs) {
    // Generate random number on GPU
    curandState* state = reinterpret_cast<curandState*>(d_rng_state_);
    float random_val;
    
    // Generate proper random number with C++11 random
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    random_val = dis(gen);
    
    // Compute cumulative sum
    float* d_cumsum;
    CUDA_CHECK(cudaMalloc(&d_cumsum, config_.vocab_size * sizeof(float)));
    
    int block_size = 256;
    int num_blocks = (config_.vocab_size + block_size - 1) / block_size;
    cumulativeSumKernel<<<num_blocks, block_size, 0, stream_>>>(
        d_probs, d_cumsum, config_.vocab_size
    );
    
    // Binary search for sample
    int* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));
    
    binarySearchSampleKernel<<<1, 1, 0, stream_>>>(
        d_cumsum, random_val, d_result, config_.vocab_size
    );
    
    int h_result;
    CUDA_CHECK(cudaMemcpyAsync(&h_result, d_result, sizeof(int),
                                cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    CUDA_CHECK(cudaFree(d_cumsum));
    CUDA_CHECK(cudaFree(d_result));
    
    return h_result;
}

int GPUDecoder::topKSampleGPU(const float* d_probs) {
    // Top-K sampling: mask out all but top-k probabilities, then sample
    // For now, fall back to temperature sampling
    // Full implementation would use thrust::sort or radix select
    return temperatureSampleGPU(d_probs);
}

int GPUDecoder::topPSampleGPU(const float* d_probs) {
    // Nucleus (top-p) sampling: mask probabilities below cumulative threshold
    // For now, fall back to temperature sampling
    // Full implementation would use cumulative sum + threshold kernel
    return temperatureSampleGPU(d_probs);
}

float* GPUDecoder::decodeGPU(const float* d_neural_output) {
    // Project to logits
    projectToLogitsGPU(d_neural_output, d_logits_);
    
    // Apply softmax
    softmaxGPU(d_logits_, d_probabilities_);
    
    return d_probabilities_;
}

int GPUDecoder::decodeAndSampleGPU(const float* d_neural_output) {
    // Decode to probabilities
    float* d_probs = decodeGPU(d_neural_output);
    
    // Sample token
    return sampleTokenGPU(d_probs);
}

std::vector<float> GPUDecoder::decode(const std::vector<float>& neural_output) {
    // Copy input to GPU
    CUDA_CHECK(cudaMemcpyAsync(d_neural_input_, neural_output.data(),
                                config_.output_dim * sizeof(float),
                                cudaMemcpyHostToDevice, stream_));
    
    // Decode on GPU
    float* d_probs = decodeGPU(d_neural_input_);
    
    // Copy result back to CPU
    CUDA_CHECK(cudaMemcpyAsync(h_probabilities_, d_probs,
                                config_.vocab_size * sizeof(float),
                                cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    return std::vector<float>(h_probabilities_, h_probabilities_ + config_.vocab_size);
}

int GPUDecoder::decodeAndSample(const std::vector<float>& neural_output) {
    // Copy input to GPU
    CUDA_CHECK(cudaMemcpyAsync(d_neural_input_, neural_output.data(),
                                config_.output_dim * sizeof(float),
                                cudaMemcpyHostToDevice, stream_));
    
    // Decode and sample on GPU
    return decodeAndSampleGPU(d_neural_input_);
}

std::pair<int, float> GPUDecoder::decodeAndSampleWithProb(const std::vector<float>& neural_output) {
    // Copy input activations to GPU workspace
    CUDA_CHECK(cudaMemcpyAsync(d_neural_input_, neural_output.data(),
                                config_.output_dim * sizeof(float),
                                cudaMemcpyHostToDevice, stream_));

    // Decode to probabilities on GPU
    float* d_probs = decodeGPU(d_neural_input_);

    // Sample a token using the configured sampling strategy
    int token_id = sampleTokenGPU(d_probs);

    // Copy full probability distribution back to host (already allocated pinned buffer)
    CUDA_CHECK(cudaMemcpyAsync(h_probabilities_, d_probs,
                                config_.vocab_size * sizeof(float),
                                cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // Extract probability of the sampled token; guard against out-of-range indices
    float prob = 0.0f;
    if (token_id >= 0 && token_id < config_.vocab_size) {
        prob = h_probabilities_[token_id];
    }

    return std::make_pair(token_id, prob);
}

std::string GPUDecoder::decodeToString(const std::vector<float>& neural_output) {
    int token_id = decodeAndSample(neural_output);
    // TokenEmbedding doesn't have decodeString, return token ID as string
    return std::to_string(token_id);
}

/**
 * @brief CUDA kernel for computing gradient and updating projection matrix
 * 
 * For cross-entropy loss with softmax, the gradient w.r.t. logits is:
 *   dL/d_logits = probs - one_hot(target)
 * 
 * The gradient w.r.t. projection matrix W is:
 *   dL/dW[i,j] = (probs[i] - one_hot[i]) * input[j]
 * 
 * This kernel updates W[i,j] -= lr * dL/dW[i,j] for one row
 */
__global__ void updateProjectionMatrixKernel(
    float* __restrict__ W,           // [vocab_size, output_dim]
    const float* __restrict__ input, // [output_dim]
    const float* __restrict__ probs, // [vocab_size]
    int target_token,                // Target token index
    int vocab_size,
    int output_dim,
    float learning_rate,
    int col_offset                   // For handling large output_dim
) {
    // Each thread handles one (row, col) pair
    int row = blockIdx.x;  // vocab token index
    int col = threadIdx.x + col_offset; // input dimension index
    
    if (row >= vocab_size || col >= output_dim) return;
    
    // Gradient for this token: prob - (1 if target else 0)
    float grad_logit = probs[row] - (row == target_token ? 1.0f : 0.0f);
    
    // Gradient for W[row, col] = grad_logit * input[col]
    float grad_W = grad_logit * input[col];
    
    // Apply gradient with learning rate (SGD update)
    int idx = row * output_dim + col;
    W[idx] -= learning_rate * grad_W;
}

/**
 * @brief Update bias vector: b -= lr * (probs - one_hot)
 */
__global__ void updateBiasKernel(
    float* __restrict__ bias,        // [vocab_size]
    const float* __restrict__ probs, // [vocab_size]
    int target_token,
    int vocab_size,
    float learning_rate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vocab_size) return;
    
    float grad = probs[idx] - (idx == target_token ? 1.0f : 0.0f);
    bias[idx] -= learning_rate * grad;
}

float GPUDecoder::trainStep(const std::vector<float>& neural_output, 
                            int target_token_id,
                            const std::vector<float>& predicted_probs) {
    if (target_token_id < 0 || target_token_id >= config_.vocab_size) {
        return 0.0f;  // Invalid target
    }
    
    // Compute cross-entropy loss: -log(p[target])
    const float eps = 1e-8f;
    float loss = -std::log(std::max(predicted_probs[target_token_id], eps));
    
    // Copy input to GPU (handle size mismatch)
    size_t input_size = std::min(neural_output.size(), static_cast<size_t>(config_.output_dim));
    CUDA_CHECK(cudaMemsetAsync(d_neural_input_, 0, config_.output_dim * sizeof(float), stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_neural_input_, neural_output.data(),
                               input_size * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    
    // Copy predicted probabilities to GPU
    CUDA_CHECK(cudaMemcpyAsync(d_probabilities_, predicted_probs.data(),
                               config_.vocab_size * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    // Update projection matrix in chunks (handles output_dim > 1024)
    int num_blocks = config_.vocab_size;
    for (int col_offset = 0; col_offset < config_.output_dim; col_offset += 1024) {
        int chunk_size = std::min(1024, config_.output_dim - col_offset);
        updateProjectionMatrixKernel<<<num_blocks, chunk_size, 0, stream_>>>(
            d_projection_matrix_,
            d_neural_input_,
            d_probabilities_,
            target_token_id,
            config_.vocab_size,
            config_.output_dim,
            config_.learning_rate,
            col_offset
        );
    }
    
    // Update bias
    int bias_blocks = (config_.vocab_size + 255) / 256;
    updateBiasKernel<<<bias_blocks, 256, 0, stream_>>>(
        d_projection_bias_,
        d_probabilities_,
        target_token_id,
        config_.vocab_size,
        config_.learning_rate
    );
    
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    return loss;
}

void GPUDecoder::saveWeights(const std::string& filepath) {
    // Allocate host memory
    size_t matrix_size = config_.vocab_size * config_.output_dim;
    std::vector<float> h_projection_matrix(matrix_size);
    std::vector<float> h_projection_bias(config_.vocab_size);
    
    // Copy from GPU to host
    CUDA_CHECK(cudaMemcpy(h_projection_matrix.data(), d_projection_matrix_,
                          matrix_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_projection_bias.data(), d_projection_bias_,
                          config_.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Write to file
    std::ofstream out(filepath, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    
    // Write header
    uint32_t magic = 0x44454344; // "DECD"
    uint32_t version = 1;
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&config_.vocab_size), sizeof(config_.vocab_size));
    out.write(reinterpret_cast<const char*>(&config_.output_dim), sizeof(config_.output_dim));
    
    // Write projection matrix
    out.write(reinterpret_cast<const char*>(h_projection_matrix.data()),
              matrix_size * sizeof(float));
    
    // Write projection bias
    out.write(reinterpret_cast<const char*>(h_projection_bias.data()),
              config_.vocab_size * sizeof(float));
    
    out.close();
    std::cout << "✓ Saved decoder weights to " << filepath << std::endl;
}

void GPUDecoder::loadWeights(const std::string& filepath) {
    // Open file
    std::ifstream in(filepath, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file for reading: " + filepath);
    }
    
    // Read and verify header
    uint32_t magic, version;
    int vocab_size, output_dim;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
    in.read(reinterpret_cast<char*>(&output_dim), sizeof(output_dim));
    
    if (magic != 0x44454344) {
        throw std::runtime_error("Invalid decoder weights file (bad magic number)");
    }
    if (vocab_size != config_.vocab_size || output_dim != config_.output_dim) {
        throw std::runtime_error("Decoder weights mismatch: expected " + 
                                std::to_string(config_.vocab_size) + "x" +
                                std::to_string(config_.output_dim) + ", got " +
                                std::to_string(vocab_size) + "x" + std::to_string(output_dim));
    }
    
    // Read projection matrix
    size_t matrix_size = config_.vocab_size * config_.output_dim;
    std::vector<float> h_projection_matrix(matrix_size);
    in.read(reinterpret_cast<char*>(h_projection_matrix.data()),
            matrix_size * sizeof(float));
    
    // Read projection bias
    std::vector<float> h_projection_bias(config_.vocab_size);
    in.read(reinterpret_cast<char*>(h_projection_bias.data()),
            config_.vocab_size * sizeof(float));
    
    in.close();
    
    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(d_projection_matrix_, h_projection_matrix.data(),
                          matrix_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_projection_bias_, h_projection_bias.data(),
                          config_.vocab_size * sizeof(float), cudaMemcpyHostToDevice));
    
    std::cout << "✓ Loaded decoder weights from " << filepath << std::endl;
}

std::vector<float> GPUDecoder::decodeToLogits(const std::vector<float>& neural_output) {
    // Copy input to GPU
    size_t input_size = std::min(neural_output.size(), static_cast<size_t>(config_.output_dim));
    CUDA_CHECK(cudaMemcpyAsync(d_neural_input_, neural_output.data(),
                               input_size * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    
    // Zero-pad if input is smaller than expected
    if (input_size < static_cast<size_t>(config_.output_dim)) {
        CUDA_CHECK(cudaMemsetAsync(d_neural_input_ + input_size, 0,
                                   (config_.output_dim - input_size) * sizeof(float),
                                   stream_));
    }
    
    // Project to logits
    projectToLogitsGPU(d_neural_input_, d_logits_);
    
    // Copy logits back to host
    std::vector<float> logits(config_.vocab_size);
    CUDA_CHECK(cudaMemcpy(logits.data(), d_logits_,
                          config_.vocab_size * sizeof(float),
                          cudaMemcpyDeviceToHost));
    
    return logits;
}

std::vector<float> GPUDecoder::getProjectionMatrix() {
    size_t matrix_size = static_cast<size_t>(config_.vocab_size) * config_.output_dim;
    std::vector<float> h_matrix(matrix_size);
    CUDA_CHECK(cudaMemcpy(h_matrix.data(), d_projection_matrix_,
                          matrix_size * sizeof(float), cudaMemcpyDeviceToHost));
    return h_matrix;
}

void GPUDecoder::setProjectionMatrix(const std::vector<float>& matrix) {
    size_t expected_size = static_cast<size_t>(config_.vocab_size) * config_.output_dim;
    if (matrix.size() != expected_size) {
        throw std::runtime_error("Projection matrix size mismatch: expected " +
                                std::to_string(expected_size) + ", got " +
                                std::to_string(matrix.size()));
    }
    CUDA_CHECK(cudaMemcpy(d_projection_matrix_, matrix.data(),
                          matrix.size() * sizeof(float), cudaMemcpyHostToDevice));
}

std::vector<float> GPUDecoder::getProjectionBias() {
    std::vector<float> h_bias(config_.vocab_size);
    CUDA_CHECK(cudaMemcpy(h_bias.data(), d_projection_bias_,
                          config_.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    return h_bias;
}

void GPUDecoder::setProjectionBias(const std::vector<float>& bias) {
    if (static_cast<int>(bias.size()) != config_.vocab_size) {
        throw std::runtime_error("Projection bias size mismatch: expected " +
                                std::to_string(config_.vocab_size) + ", got " +
                                std::to_string(bias.size()));
    }
    CUDA_CHECK(cudaMemcpy(d_projection_bias_, bias.data(),
                          bias.size() * sizeof(float), cudaMemcpyHostToDevice));
}

