#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

// CUDA header inclusion with conflict resolution
// Strategic CUDA header inclusion to avoid conflicts
#ifndef __CUDACC__
    // Host compilation - minimal CUDA headers
    #include <cuda_runtime.h>
#else
    // Device compilation - full CUDA headers with proper ordering
    #include <cuda_runtime.h>
    
    // Include curand BEFORE device_launch_parameters to avoid conflicts
    #include <curand_kernel.h>
    #include <device_launch_parameters.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <stdexcept>

// Include compatibility header for additional utilities
#include "CudaCompatibility.h"

// Forward declarations for GPU structures (avoid circular dependencies)
struct GPUNeuronState;
struct GPUSynapse;
struct GPUSpikeEvent;

// ============================================================================
// CUDA ERROR HANDLING MACROS
// ============================================================================

/**
 * @brief Comprehensive CUDA error checking with detailed diagnostics
 */
#define CUDA_CHECK_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::string error_msg = std::string("CUDA Error: ") + \
                                  cudaGetErrorString(err) + \
                                  " at " + __FILE__ + ":" + std::to_string(__LINE__) + \
                                  " in function '" + __func__ + "'"; \
            std::cerr << error_msg << std::endl; \
            throw std::runtime_error(error_msg); \
        } \
    } while(0)

/**
 * @brief Check for CUDA kernel launch errors and synchronize
 */
#define CUDA_KERNEL_CHECK() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::string error_msg = std::string("CUDA Kernel Error: ") + \
                                  cudaGetErrorString(err) + \
                                  " at " + __FILE__ + ":" + std::to_string(__LINE__); \
            std::cerr << error_msg << std::endl; \
            throw std::runtime_error(error_msg); \
        } \
        CUDA_CHECK_ERROR(cudaDeviceSynchronize()); \
    } while(0)

/**
 * @brief Lightweight kernel error check without synchronization
 */
#define CUDA_KERNEL_CHECK_ASYNC() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::string error_msg = std::string("CUDA Kernel Error (Async): ") + \
                                  cudaGetErrorString(err) + \
                                  " at " + __FILE__ + ":" + std::to_string(__LINE__); \
            std::cerr << error_msg << std::endl; \
            throw std::runtime_error(error_msg); \
        } \
    } while(0)

// ============================================================================
// CUDA GRID AND BLOCK UTILITIES
// ============================================================================

namespace CudaUtils {
    
    // Constants for optimal kernel configuration
    constexpr int DEFAULT_BLOCK_SIZE = 256;
    constexpr int MAX_BLOCKS = 65535;
    constexpr int WARP_SIZE = 32;
    constexpr int MAX_THREADS_PER_BLOCK = 1024;
    
    /**
     * @brief Create optimized CUDA block dimensions
     * @param size Desired block size (will be adjusted to valid range)
     * @return Optimal dim3 block configuration
     */
    inline __host__ dim3 makeSafeBlock(int size = DEFAULT_BLOCK_SIZE) {
        // Ensure block size is multiple of warp size for efficiency
        size = ((size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
        
        // Clamp to valid range
        if (size <= 0) size = DEFAULT_BLOCK_SIZE;
        if (size > MAX_THREADS_PER_BLOCK) size = MAX_THREADS_PER_BLOCK;
        
        return dim3(static_cast<unsigned int>(size));
    }
    
    /**
     * @brief Create optimized CUDA grid dimensions
     * @param total_threads Total number of threads needed
     * @param block_size Threads per block
     * @return Optimal dim3 grid configuration
     */
    inline __host__ dim3 makeSafeGrid(int total_threads, int block_size = DEFAULT_BLOCK_SIZE) {
        if (total_threads <= 0) return dim3(1);
        if (block_size <= 0) block_size = DEFAULT_BLOCK_SIZE;
        
        int grid_size = (total_threads + block_size - 1) / block_size;
        
        // Handle large grids by using 2D configuration if needed
        if (grid_size <= MAX_BLOCKS) {
            return dim3(static_cast<unsigned int>(grid_size));
        } else {
            // Use 2D grid for very large workloads
            int grid_x = MAX_BLOCKS;
            int grid_y = (grid_size + MAX_BLOCKS - 1) / MAX_BLOCKS;
            return dim3(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y));
        }
    }
    
    /**
     * @brief Get optimal block size for a given workload
     * @param num_threads Total number of threads
     * @return Recommended block size
     */
    inline __host__ int getOptimalBlockSize(int num_threads) {
        if (num_threads >= 1024) return 1024;
        if (num_threads >= 512) return 512;
        if (num_threads >= 256) return 256;
        if (num_threads >= 128) return 128;
        if (num_threads >= 64) return 64;
        return 32; // Minimum warp size
    }
    
    /**
     * @brief Calculate 2D grid configuration for matrix operations
     * @param width Matrix width
     * @param height Matrix height
     * @param block_size Block size for both dimensions
     * @return Grid and block configuration pair
     */
    inline __host__ std::pair<dim3, dim3> make2DConfig(int width, int height, int block_size = 16) {
        dim3 block(block_size, block_size);
        dim3 grid((width + block_size - 1) / block_size,
                  (height + block_size - 1) / block_size);
        return std::make_pair(grid, block);
    }
    
    /**
     * @brief Get CUDA device properties and print information
     * @param device_id CUDA device ID (default: 0)
     */
    inline void printDeviceInfo(int device_id = 0) {
        cudaDeviceProp prop;
        CUDA_CHECK_ERROR(cudaGetDeviceProperties(&prop, device_id));
        
        std::cout << "=== CUDA Device Information ===" << std::endl;
        std::cout << "Device: " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Total Global Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
        std::cout << "Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Max Block Dimensions: " << prop.maxThreadsDim[0] << " x " 
                  << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << std::endl;
        std::cout << "Max Grid Dimensions: " << prop.maxGridSize[0] << " x " 
                  << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << std::endl;
        std::cout << "Warp Size: " << prop.warpSize << std::endl;
        std::cout << "Memory Clock Rate: " << (prop.memoryClockRate / 1000) << " MHz" << std::endl;
        std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "===============================" << std::endl;
    }
    
    /**
     * @brief Check available GPU memory
     * @return Pair of (free_bytes, total_bytes)
     */
    inline std::pair<size_t, size_t> getMemoryInfo() {
        size_t free_mem, total_mem;
        CUDA_CHECK_ERROR(cudaMemGetInfo(&free_mem, &total_mem));
        return std::make_pair(free_mem, total_mem);
    }
    
    /**
     * @brief Set optimal CUDA device for neural network operations
     * @return Selected device ID
     */
    inline int selectOptimalDevice() {
        int device_count;
        CUDA_CHECK_ERROR(cudaGetDeviceCount(&device_count));
        
        if (device_count == 0) {
            throw std::runtime_error("No CUDA-capable devices found");
        }
        
        int best_device = 0;
        size_t max_memory = 0;
        
        for (int i = 0; i < device_count; ++i) {
            cudaDeviceProp prop;
            CUDA_CHECK_ERROR(cudaGetDeviceProperties(&prop, i));
            
            // Prefer devices with more memory and higher compute capability
            size_t device_score = prop.totalGlobalMem + 
                                 (static_cast<size_t>(prop.major * 10 + prop.minor) * 1024 * 1024 * 1024);
            
            if (device_score > max_memory) {
                max_memory = device_score;
                best_device = i;
            }
        }
        
        CUDA_CHECK_ERROR(cudaSetDevice(best_device));
        std::cout << "Selected CUDA device " << best_device << " for neural network processing" << std::endl;
        
        return best_device;
    }

} // namespace CudaUtils

// ============================================================================
// DEVICE FUNCTION UTILITIES
// ============================================================================

#ifdef __CUDACC__

/**
 * @brief Thread-safe atomic addition for floats (if not natively supported)
 */
__device__ inline float atomicAddFloat(float* address, float val) {
    #if __CUDA_ARCH__ >= 200
        return atomicAdd(address, val);
    #else
        // Fallback implementation for older architectures
        int* address_as_int = (int*)address;
        int old = *address_as_int, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_int, assumed,
                          __float_as_int(val + __int_as_float(assumed)));
        } while (assumed != old);
        return __int_as_float(old);
    #endif
}

/**
 * @brief Fast device-side random number generation
 */
__device__ inline float fastRandom(unsigned int& seed) {
    seed = (seed * 1664525u + 1013904223u);
    return static_cast<float>(seed) / 4294967296.0f;
}

/**
 * @brief Device-side sigmoid activation function
 */
__device__ inline float sigmoidDevice(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/**
 * @brief Device-side ReLU activation function
 */
__device__ inline float reluDevice(float x) {
    return fmaxf(0.0f, x);
}

/**
 * @brief Device-side tanh activation function (optimized)
 */
__device__ inline float tanhDevice(float x) {
    return tanhf(x);
}

/**
 * @brief Clamp value to range [min_val, max_val]
 */
__device__ inline float clampDevice(float x, float min_val, float max_val) {
    return fminf(fmaxf(x, min_val), max_val);
}

#endif // __CUDACC__

// ============================================================================
// MEMORY MANAGEMENT UTILITIES
// ============================================================================

namespace CudaMemory {
    
    /**
     * @brief RAII wrapper for CUDA device memory
     */
    template<typename T>
    class DeviceArray {
    private:
        T* ptr_;
        size_t size_;
        
    public:
        explicit DeviceArray(size_t count) : size_(count) {
            CUDA_CHECK_ERROR(cudaMalloc(&ptr_, count * sizeof(T)));
        }
        
        ~DeviceArray() {
            if (ptr_) {
                cudaFree(ptr_);
            }
        }
        
        // Delete copy constructor and assignment
        DeviceArray(const DeviceArray&) = delete;
        DeviceArray& operator=(const DeviceArray&) = delete;
        
        // Move constructor and assignment
        DeviceArray(DeviceArray&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        
        DeviceArray& operator=(DeviceArray&& other) noexcept {
            if (this != &other) {
                if (ptr_) cudaFree(ptr_);
                ptr_ = other.ptr_;
                size_ = other.size_;
                other.ptr_ = nullptr;
                other.size_ = 0;
            }
            return *this;
        }
        
        T* get() const { return ptr_; }
        size_t size() const { return size_; }
        
        void copyFromHost(const T* host_data, size_t count = 0) {
            if (count == 0) count = size_;
            CUDA_CHECK_ERROR(cudaMemcpy(ptr_, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
        }
        
        void copyToHost(T* host_data, size_t count = 0) const {
            if (count == 0) count = size_;
            CUDA_CHECK_ERROR(cudaMemcpy(host_data, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
        }
        
        void zero() {
            CUDA_CHECK_ERROR(cudaMemset(ptr_, 0, size_ * sizeof(T)));
        }
    };
    
    /**
     * @brief Allocate pinned host memory for faster transfers
     */
    template<typename T>
    class PinnedArray {
    private:
        T* ptr_;
        size_t size_;
        
    public:
        explicit PinnedArray(size_t count) : size_(count) {
            CUDA_CHECK_ERROR(cudaMallocHost(&ptr_, count * sizeof(T)));
        }
        
        ~PinnedArray() {
            if (ptr_) {
                cudaFreeHost(ptr_);
            }
        }
        
        T* get() const { return ptr_; }
        size_t size() const { return size_; }
        T& operator[](size_t index) { return ptr_[index]; }
        const T& operator[](size_t index) const { return ptr_[index]; }
    };

} // namespace CudaMemory

#endif // CUDA_UTILS_H