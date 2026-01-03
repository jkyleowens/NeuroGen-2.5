#ifndef CUDA_COMPATIBILITY_H
#define CUDA_COMPATIBILITY_H

/**
 * @brief Clean CUDA compatibility layer for breakthrough neural networks
 * 
 * This header provides a conflict-free interface to CUDA functionality
 * while maintaining the performance critical for brain-like processing.
 */

// Essential CUDA runtime only - avoiding conflicting headers
#include <cuda_runtime.h>

// Standard library includes
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

// ============================================================================
// CUDA KERNEL EXECUTION MACROS WITHOUT CONFLICTS
// ============================================================================

/**
 * @brief Safe CUDA error checking with detailed neural network context
 */
#define NEURAL_CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::string error_msg = std::string("Neural CUDA Error: ") + \
                                  cudaGetErrorString(err) + \
                                  " at " + __FILE__ + ":" + std::to_string(__LINE__) + \
                                  " in breakthrough neural function '" + __func__ + "'"; \
            std::cerr << "[NEURAL ERROR] " << error_msg << std::endl; \
            throw std::runtime_error(error_msg); \
        } \
    } while(0)

/**
 * @brief Neural network kernel launch error checking
 */
#define NEURAL_KERNEL_CHECK() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::string error_msg = std::string("Neural Kernel Error: ") + \
                                  cudaGetErrorString(err) + \
                                  " in brain-mimicking processing at " + __FILE__ + ":" + std::to_string(__LINE__); \
            std::cerr << "[NEURAL KERNEL ERROR] " << error_msg << std::endl; \
            throw std::runtime_error(error_msg); \
        } \
        NEURAL_CUDA_CHECK(cudaDeviceSynchronize()); \
    } while(0)

/**
 * @brief Async kernel check for pipeline optimization
 */
#define NEURAL_KERNEL_CHECK_ASYNC() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::string error_msg = std::string("Neural Kernel Error (Async): ") + \
                                  cudaGetErrorString(err) + \
                                  " in modular neural processing"; \
            std::cerr << "[NEURAL ASYNC ERROR] " << error_msg << std::endl; \
            throw std::runtime_error(error_msg); \
        } \
    } while(0)

// ============================================================================
// NEURAL NETWORK SPECIFIC CUDA UTILITIES
// ============================================================================

namespace NeuralCuda {
    
    /**
     * @brief Constants optimized for neural network processing
     */
    constexpr int NEURAL_BLOCK_SIZE = 256;      // Optimal for most neural kernels
    constexpr int SYNAPTIC_BLOCK_SIZE = 512;    // Optimized for synaptic processing
    constexpr int COLUMN_BLOCK_SIZE = 64;       // Perfect for cortical columns
    constexpr int MAX_GRID_SIZE = 65535;        // CUDA grid limitation
    constexpr int WARP_SIZE = 32;               // GPU warp size
    
    /**
     * @brief Create optimal block configuration for neural processing
     */
    inline dim3 makeNeuralBlock(int neurons) {
        int block_size = (neurons < NEURAL_BLOCK_SIZE) ? neurons : NEURAL_BLOCK_SIZE;
        // Ensure warp alignment for efficiency
        block_size = ((block_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
        return dim3(static_cast<unsigned int>(block_size));
    }
    
    /**
     * @brief Create optimal grid configuration for neural processing
     */
    inline dim3 makeNeuralGrid(int total_neurons, int block_size = NEURAL_BLOCK_SIZE) {
        if (total_neurons <= 0) return dim3(1);
        
        int grid_size = (total_neurons + block_size - 1) / block_size;
        
        if (grid_size <= MAX_GRID_SIZE) {
            return dim3(static_cast<unsigned int>(grid_size));
        } else {
            // Use 2D grid for very large neural networks
            int grid_x = MAX_GRID_SIZE;
            int grid_y = (grid_size + MAX_GRID_SIZE - 1) / MAX_GRID_SIZE;
            return dim3(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y));
        }
    }
    
    /**
     * @brief Specialized configuration for synaptic processing
     */
    inline std::pair<dim3, dim3> makeSynapticConfig(int num_synapses) {
        dim3 block(SYNAPTIC_BLOCK_SIZE);
        dim3 grid = makeNeuralGrid(num_synapses, SYNAPTIC_BLOCK_SIZE);
        return std::make_pair(grid, block);
    }
    
    /**
     * @brief Specialized configuration for cortical column processing
     */
    inline std::pair<dim3, dim3> makeColumnConfig(int num_columns) {
        dim3 block(COLUMN_BLOCK_SIZE);
        dim3 grid = makeNeuralGrid(num_columns, COLUMN_BLOCK_SIZE);
        return std::make_pair(grid, block);
    }
    
    /**
     * @brief Get neural network optimized device properties
     */
    inline void printNeuralDeviceInfo(int device_id = 0) {
        cudaDeviceProp prop;
        NEURAL_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
        
        std::cout << "=== Breakthrough Neural Processing Device ===" << std::endl;
        std::cout << "Device: " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Neural Memory Capacity: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
        std::cout << "Neural Processors: " << prop.multiProcessorCount << std::endl;
        std::cout << "Max Neural Threads/Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Neural Memory Bandwidth: " << (prop.memoryBusWidth / 8 * prop.memoryClockRate * 2 / 1024 / 1024) << " GB/s" << std::endl;
        std::cout << "Brain-Like Processing Ready: " << (prop.major >= 6 ? "YES" : "LIMITED") << std::endl;
        std::cout << "=============================================" << std::endl;
    }
    
    /**
     * @brief Check neural network memory requirements
     */
    inline std::pair<size_t, size_t> getNeuralMemoryInfo() {
        size_t free_mem, total_mem;
        NEURAL_CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        
        std::cout << "[NEURAL MEMORY] Available: " << (free_mem / 1024 / 1024) 
                  << " MB / Total: " << (total_mem / 1024 / 1024) << " MB" << std::endl;
        
        return std::make_pair(free_mem, total_mem);
    }
    
    /**
     * @brief Select optimal GPU for breakthrough neural processing
     */
    inline int selectNeuralDevice() {
        int device_count;
        NEURAL_CUDA_CHECK(cudaGetDeviceCount(&device_count));
        
        if (device_count == 0) {
            throw std::runtime_error("[NEURAL ERROR] No CUDA devices found for breakthrough processing");
        }
        
        int best_device = 0;
        int best_score = 0;
        
        for (int i = 0; i < device_count; ++i) {
            cudaDeviceProp prop;
            NEURAL_CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
            
            // Score based on neural network requirements
            int score = prop.multiProcessorCount * 100 +           // Processing units
                       (prop.totalGlobalMem / (1024*1024)) / 100 + // Memory capacity
                       prop.major * 1000 +                        // Compute capability
                       (prop.memoryBusWidth / 32);                 // Memory bandwidth
            
            if (score > best_score) {
                best_score = score;
                best_device = i;
            }
        }
        
        NEURAL_CUDA_CHECK(cudaSetDevice(best_device));
        
        cudaDeviceProp prop;
        NEURAL_CUDA_CHECK(cudaGetDeviceProperties(&prop, best_device));
        
        std::cout << "[NEURAL SYSTEM] Selected optimal GPU " << best_device 
                  << " (" << prop.name << ") for breakthrough brain processing" << std::endl;
        
        return best_device;
    }
    
    /**
     * @brief Validate neural kernel launch parameters
     */
    inline bool validateNeuralLaunch(dim3 grid, dim3 block, int expected_elements) {
        // Check thread limits
        if (block.x * block.y * block.z > 1024) {
            std::cerr << "[NEURAL WARNING] Block size exceeds GPU limits" << std::endl;
            return false;
        }
        
        // Check grid limits
        if (grid.x > MAX_GRID_SIZE || grid.y > MAX_GRID_SIZE) {
            std::cerr << "[NEURAL WARNING] Grid size exceeds GPU limits" << std::endl;
            return false;
        }
        
        // Check coverage
        int total_threads = grid.x * grid.y * grid.z * block.x * block.y * block.z;
        if (total_threads < expected_elements) {
            std::cerr << "[NEURAL WARNING] Insufficient thread coverage for neural elements" << std::endl;
            return false;
        }
        
        return true;
    }
    
    /**
     * @brief Calculate optimal shared memory for neural kernels
     */
    inline size_t calculateNeuralSharedMemory(int threads_per_block, size_t memory_per_thread) {
        size_t total_required = threads_per_block * memory_per_thread;
        
        // Typical GPU shared memory limits
        constexpr size_t MAX_SHARED_MEMORY = 49152; // 48KB
        
        if (total_required > MAX_SHARED_MEMORY) {
            std::cout << "[NEURAL INFO] Shared memory requirement (" << total_required 
                      << " bytes) exceeds limit, using maximum available" << std::endl;
            return MAX_SHARED_MEMORY;
        }
        
        return total_required;
    }

} // namespace NeuralCuda

// ============================================================================
// DEVICE FUNCTION DECORATORS (CONFLICT-FREE)
// ============================================================================

#ifdef __CUDACC__
    #define NEURAL_DEVICE __device__
    #define NEURAL_HOST __host__
    #define NEURAL_GLOBAL __global__
    #define NEURAL_HOST_DEVICE __host__ __device__
    #define NEURAL_INLINE __forceinline__
#else
    #define NEURAL_DEVICE
    #define NEURAL_HOST
    #define NEURAL_GLOBAL
    #define NEURAL_HOST_DEVICE
    #define NEURAL_INLINE inline
#endif

// ============================================================================
// NEURAL-SPECIFIC MATHEMATICAL CONSTANTS
// ============================================================================

#ifndef M_PI_NEURAL
#define M_PI_NEURAL 3.14159265358979323846f
#endif

#ifndef M_E_NEURAL
#define M_E_NEURAL 2.71828182845904523536f
#endif

// Biologically relevant constants
constexpr float RESTING_POTENTIAL = -70.0f;     // mV
constexpr float SPIKE_THRESHOLD = -55.0f;       // mV
constexpr float RESET_POTENTIAL = -70.0f;       // mV
constexpr float MEMBRANE_TAU = 20.0f;           // ms
constexpr float SYNAPTIC_TAU = 5.0f;            // ms
constexpr float REFACTORY_PERIOD = 2.0f;        // ms

// ============================================================================
// NEURAL NETWORK OPTIMIZED DEVICE FUNCTIONS
// ============================================================================

#ifdef __CUDACC__

/**
 * @brief Fast neural activation functions optimized for GPU
 */
NEURAL_DEVICE NEURAL_INLINE float neuralSigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

NEURAL_DEVICE NEURAL_INLINE float neuralTanh(float x) {
    return tanhf(x);
}

NEURAL_DEVICE NEURAL_INLINE float neuralReLU(float x) {
    return fmaxf(0.0f, x);
}

NEURAL_DEVICE NEURAL_INLINE float neuralLeakyReLU(float x, float alpha = 0.01f) {
    return (x > 0.0f) ? x : alpha * x;
}

/**
 * @brief Biologically accurate membrane dynamics
 */
NEURAL_DEVICE NEURAL_INLINE float integrateAndFire(float voltage, float input_current, float dt) {
    float leak_current = -(voltage - RESTING_POTENTIAL) / MEMBRANE_TAU;
    return voltage + (leak_current + input_current) * dt;
}

/**
 * @brief Synaptic transmission with realistic dynamics
 */
NEURAL_DEVICE NEURAL_INLINE float synapticResponse(float pre_spike_time, float current_time, float tau = SYNAPTIC_TAU) {
    float dt = current_time - pre_spike_time;
    return (dt > 0.0f) ? expf(-dt / tau) : 0.0f;
}

/**
 * @brief Neural noise generation for biological realism
 */
NEURAL_DEVICE NEURAL_INLINE float neuralNoise(unsigned int& seed, float amplitude = 0.1f) {
    seed = seed * 1664525u + 1013904223u;
    float uniform = static_cast<float>(seed) / 4294967296.0f;
    return amplitude * (uniform - 0.5f) * 2.0f;
}

/**
 * @brief Spike-timing dependent plasticity calculation
 */
NEURAL_DEVICE NEURAL_INLINE float calculateSTDP(float pre_time, float post_time, 
                                                float tau_plus = 20.0f, float tau_minus = 20.0f,
                                                float A_plus = 0.01f, float A_minus = 0.01f) {
    float dt = post_time - pre_time;
    if (dt > 0.0f) {
        return A_plus * expf(-dt / tau_plus);
    } else {
        return -A_minus * expf(dt / tau_minus);
    }
}

/**
 * @brief Clamp values to biological ranges
 */
NEURAL_DEVICE NEURAL_INLINE float neuralClamp(float value, float min_val, float max_val) {
    return fminf(fmaxf(value, min_val), max_val);
}

/**
 * @brief Distance calculation for spatial neural networks
 */
NEURAL_DEVICE NEURAL_INLINE float neuralDistance2D(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx * dx + dy * dy);
}

/**
 * @brief Gaussian connectivity probability
 */
NEURAL_DEVICE NEURAL_INLINE float gaussianConnectivity(float distance, float sigma, float amplitude = 1.0f) {
    float normalized_dist = distance / sigma;
    return amplitude * expf(-0.5f * normalized_dist * normalized_dist);
}

#endif // __CUDACC__

// ============================================================================
// NEURAL MEMORY MANAGEMENT UTILITIES
// ============================================================================

namespace NeuralMemory {
    
    /**
     * @brief RAII wrapper for neural network GPU memory
     */
    template<typename T>
    class NeuralDeviceArray {
    private:
        T* ptr_;
        size_t size_;
        std::string debug_name_;
        
    public:
        explicit NeuralDeviceArray(size_t count, const std::string& name = "neural_array") 
            : size_(count), debug_name_(name) {
            NEURAL_CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
            std::cout << "[NEURAL ALLOC] " << debug_name_ << ": " 
                      << (count * sizeof(T) / 1024 / 1024) << " MB" << std::endl;
        }
        
        ~NeuralDeviceArray() {
            if (ptr_) {
                cudaFree(ptr_);
                std::cout << "[NEURAL FREE] " << debug_name_ << " released" << std::endl;
            }
        }
        
        // Disable copy, enable move
        NeuralDeviceArray(const NeuralDeviceArray&) = delete;
        NeuralDeviceArray& operator=(const NeuralDeviceArray&) = delete;
        
        NeuralDeviceArray(NeuralDeviceArray&& other) noexcept 
            : ptr_(other.ptr_), size_(other.size_), debug_name_(std::move(other.debug_name_)) {
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        
        NeuralDeviceArray& operator=(NeuralDeviceArray&& other) noexcept {
            if (this != &other) {
                if (ptr_) cudaFree(ptr_);
                ptr_ = other.ptr_;
                size_ = other.size_;
                debug_name_ = std::move(other.debug_name_);
                other.ptr_ = nullptr;
                other.size_ = 0;
            }
            return *this;
        }
        
        T* get() const { return ptr_; }
        size_t size() const { return size_; }
        
        void copyFromHost(const T* host_data, size_t count = 0) {
            if (count == 0) count = size_;
            NEURAL_CUDA_CHECK(cudaMemcpy(ptr_, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
        }
        
        void copyToHost(T* host_data, size_t count = 0) const {
            if (count == 0) count = size_;
            NEURAL_CUDA_CHECK(cudaMemcpy(host_data, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
        }
        
        void zero() {
            NEURAL_CUDA_CHECK(cudaMemset(ptr_, 0, size_ * sizeof(T)));
        }
        
        void fill(const T& value) {
            // For complex types, use kernel-based filling
            if constexpr (std::is_arithmetic_v<T> && sizeof(T) <= 4) {
                if constexpr (std::is_same_v<T, float>) {
                    // Use cudaMemset for simple types
                    if (value == 0.0f) {
                        zero();
                        return;
                    }
                }
            }
            
            // Fallback: copy pattern from host
            std::vector<T> pattern(size_, value);
            copyFromHost(pattern.data());
        }
    };
    
} // namespace NeuralMemory

#endif // CUDA_COMPATIBILITY_H