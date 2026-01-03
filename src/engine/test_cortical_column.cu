/**
 * @file test_cortical_column.cu
 * @brief Test program for CorticalColumnV2 architecture
 * 
 * Validates:
 * - ALIF neuron updates
 * - Sparse synaptic propagation
 * - Connectivity generation
 * - Basic column simulation
 * 
 * @version 2.0-Cortical
 * @date November 29, 2025
 */

#include "engine/CorticalColumnV2.h"
#include "engine/ALIFKernels.cuh"
#include "engine/SparseKernels.cuh"
#include "engine/STDPKernels.cuh"
#include "engine/ConnectivityGenerator.h"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

using namespace neurogen::cortical;

// Test basic ALIF neuron functionality
bool testALIFNeurons() {
    std::cout << "Testing ALIF neurons..." << std::endl;
    
    const int num_neurons = 1000;
    
    ALIFNeuronArrays neurons;
    neurons.params = ALIFParameters();  // Use defaults
    
    cudaError_t err = neurons.allocate(num_neurons);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate neurons: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    err = neurons.initialize();
    if (err != cudaSuccess) {
        std::cerr << "Failed to initialize neurons: " << cudaGetErrorString(err) << std::endl;
        neurons.free();
        return false;
    }
    
    // Inject some current to make neurons spike
    std::vector<float> currents(num_neurons, 2.0f);  // Strong input current
    cudaMemcpy(neurons.d_current, currents.data(), num_neurons * sizeof(float), cudaMemcpyHostToDevice);
    
    // Run 100 timesteps
    float dt = 1.0f;  // 1ms timestep
    for (int t = 0; t < 100; ++t) {
        err = kernels::launchALIFUpdate(neurons, dt, t * dt);
        if (err != cudaSuccess) {
            std::cerr << "ALIF update failed at t=" << t << std::endl;
            neurons.free();
            return false;
        }
    }
    
    cudaDeviceSynchronize();
    
    // Check for spikes
    std::vector<float> spikes(num_neurons);
    cudaMemcpy(spikes.data(), neurons.d_spike_output, num_neurons * sizeof(float), cudaMemcpyDeviceToHost);
    
    int spike_count = 0;
    for (int i = 0; i < num_neurons; ++i) {
        if (spikes[i] > 0.0f) {
            spike_count++;
        }
    }
    
    std::cout << "  Spike count after 100ms: " << spike_count << "/" << num_neurons << std::endl;
    
    neurons.free();
    
    return spike_count > 0;  // At least some neurons should spike
}

// Test sparse matrix generation and SpMV
bool testSparseSynapses() {
    std::cout << "Testing sparse synapses..." << std::endl;
    
    const int num_pre = 500;
    const int num_post = 500;
    const float probability = 0.1f;
    
    CSRSynapseMatrix synapses;
    
    cudaError_t err = generateRandomConnectivity(
        synapses, num_pre, num_post, probability, 0.1f, 0.02f, 42
    );
    
    if (err != cudaSuccess) {
        std::cerr << "Failed to generate connectivity: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    std::cout << "  Generated " << synapses.nnz << " synapses" << std::endl;
    std::cout << "  Expected ~" << (int)(num_pre * num_post * probability) << " synapses" << std::endl;
    
    // Create spike input
    std::vector<uint8_t> pre_spikes(num_pre, 0);
    for (int i = 0; i < num_pre / 10; ++i) {
        pre_spikes[i * 10] = 1;  // Every 10th neuron spikes
    }
    
    uint8_t* d_pre_spikes;
    float* d_post_currents;
    
    cudaMalloc(&d_pre_spikes, num_pre * sizeof(uint8_t));
    cudaMalloc(&d_post_currents, num_post * sizeof(float));
    
    cudaMemcpy(d_pre_spikes, pre_spikes.data(), num_pre * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemset(d_post_currents, 0, num_post * sizeof(float));
    
    // Perform SpMV
    err = kernels::launchCSRSpMV(synapses, d_pre_spikes, d_post_currents, 1.0f);
    if (err != cudaSuccess) {
        std::cerr << "SpMV failed: " << cudaGetErrorString(err) << std::endl;
        synapses.free();
        cudaFree(d_pre_spikes);
        cudaFree(d_post_currents);
        return false;
    }
    
    cudaDeviceSynchronize();
    
    // Check output
    std::vector<float> post_currents(num_post);
    cudaMemcpy(post_currents.data(), d_post_currents, num_post * sizeof(float), cudaMemcpyDeviceToHost);
    
    float total_current = 0.0f;
    int nonzero_count = 0;
    for (int i = 0; i < num_post; ++i) {
        total_current += post_currents[i];
        if (post_currents[i] > 0.0f) nonzero_count++;
    }
    
    std::cout << "  Total postsynaptic current: " << total_current << std::endl;
    std::cout << "  Postsynaptic neurons with input: " << nonzero_count << "/" << num_post << std::endl;
    
    synapses.free();
    cudaFree(d_pre_spikes);
    cudaFree(d_post_currents);
    
    return nonzero_count > 0;
}

// Test cortical column
bool testCorticalColumn() {
    std::cout << "Testing cortical column..." << std::endl;
    
    CorticalColumnConfig config;
    config.name = "TestColumn";
    config.total_neurons = 1000;  // Small for testing
    config.use_compartmental = false;  // Simpler for testing
    config.enable_stdp = true;
    config.enable_homeostasis = true;
    config.global_learning_rate = 0.01f;
    
    CorticalColumnV2 column(config);
    
    cudaError_t err = column.initialize();
    if (err != cudaSuccess) {
        std::cerr << "Failed to initialize column: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    err = column.generateConnectivity();
    if (err != cudaSuccess) {
        std::cerr << "Failed to generate connectivity: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    std::cout << "  Total neurons: " << column.getTotalNeurons() << std::endl;
    std::cout << "  Total synapses: " << column.getTotalSynapses() << std::endl;
    std::cout << "  Memory footprint: " << column.getMemoryFootprint() / 1024 << " KB" << std::endl;
    
    // Inject input to L4
    int l4_size = column.getNeuronsInLayer(CorticalLayer::L4);
    std::vector<float> input(l4_size, 1.0f);  // Constant input
    
    float* d_input;
    cudaMalloc(&d_input, l4_size * sizeof(float));
    cudaMemcpy(d_input, input.data(), l4_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Run simulation
    float dt = 1.0f;  // 1ms timestep
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < 100; ++t) {
        column.injectFeedforwardInput(d_input, l4_size);
        err = column.step(dt, t * dt);
        if (err != cudaSuccess) {
            std::cerr << "Column step failed at t=" << t << std::endl;
            cudaFree(d_input);
            return false;
        }
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "  100ms simulation time: " << duration << "ms" << std::endl;
    std::cout << "  Real-time factor: " << 100.0 / duration << "x" << std::endl;
    
    // Apply STDP
    err = column.applySTDP(1.0f);  // Positive reward
    if (err != cudaSuccess) {
        std::cerr << "STDP failed" << std::endl;
    }
    
    cudaFree(d_input);
    
    return true;
}

// Test small-world connectivity
bool testSmallWorldConnectivity() {
    std::cout << "Testing small-world connectivity..." << std::endl;
    
    const int num_neurons = 500;
    const int k_neighbors = 10;
    const float rewire_prob = 0.1f;
    
    CSRSynapseMatrix synapses;
    
    cudaError_t err = generateSmallWorldConnectivity(
        synapses, num_neurons, num_neurons, k_neighbors, rewire_prob, 0.1f, 0.02f, 42
    );
    
    if (err != cudaSuccess) {
        std::cerr << "Failed to generate small-world connectivity: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    std::cout << "  Generated " << synapses.nnz << " synapses" << std::endl;
    std::cout << "  Expected ~" << (k_neighbors * num_neurons) << " synapses" << std::endl;
    
    synapses.free();
    return true;
}

// Benchmark ALIF performance
void benchmarkALIF() {
    std::cout << "\nBenchmarking ALIF performance..." << std::endl;
    
    std::vector<int> sizes = {1000, 10000, 100000, 1000000};
    
    for (int num_neurons : sizes) {
        ALIFNeuronArrays neurons;
        neurons.params = ALIFParameters();
        
        cudaError_t err = neurons.allocate(num_neurons);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate " << num_neurons << " neurons" << std::endl;
            continue;
        }
        
        neurons.initialize();
        
        // Warmup
        for (int i = 0; i < 10; ++i) {
            kernels::launchALIFUpdate(neurons, 1.0f, i);
        }
        cudaDeviceSynchronize();
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        const int iterations = 1000;
        
        for (int i = 0; i < iterations; ++i) {
            kernels::launchALIFUpdate(neurons, 1.0f, i);
        }
        cudaDeviceSynchronize();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        float ms_per_step = duration / 1000.0f / iterations;
        float neurons_per_second = num_neurons / (ms_per_step / 1000.0f);
        
        std::cout << "  " << num_neurons << " neurons: " 
                  << ms_per_step << " ms/step, "
                  << neurons_per_second / 1e6 << " M neurons/sec" << std::endl;
        
        neurons.free();
    }
}

int main() {
    std::cout << "=== CorticalColumnV2 Test Suite ===" << std::endl;
    std::cout << std::endl;
    
    // Check CUDA
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << std::endl;
    
    // Run tests
    bool all_passed = true;
    
    if (!testALIFNeurons()) {
        std::cerr << "ALIF neuron test FAILED!" << std::endl;
        all_passed = false;
    } else {
        std::cout << "ALIF neuron test PASSED!" << std::endl;
    }
    std::cout << std::endl;
    
    if (!testSparseSynapses()) {
        std::cerr << "Sparse synapse test FAILED!" << std::endl;
        all_passed = false;
    } else {
        std::cout << "Sparse synapse test PASSED!" << std::endl;
    }
    std::cout << std::endl;
    
    if (!testSmallWorldConnectivity()) {
        std::cerr << "Small-world connectivity test FAILED!" << std::endl;
        all_passed = false;
    } else {
        std::cout << "Small-world connectivity test PASSED!" << std::endl;
    }
    std::cout << std::endl;
    
    if (!testCorticalColumn()) {
        std::cerr << "Cortical column test FAILED!" << std::endl;
        all_passed = false;
    } else {
        std::cout << "Cortical column test PASSED!" << std::endl;
    }
    
    // Performance benchmarks
    benchmarkALIF();
    
    std::cout << std::endl;
    if (all_passed) {
        std::cout << "=== ALL TESTS PASSED ===" << std::endl;
        return 0;
    } else {
        std::cout << "=== SOME TESTS FAILED ===" << std::endl;
        return 1;
    }
}
