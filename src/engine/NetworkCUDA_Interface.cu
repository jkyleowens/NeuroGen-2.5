#include <engine/NetworkCUDA_Interface.h>
#include <cuda_runtime.h>
#include <iostream>

// --- Constructor ---
// Initializes the underlying NetworkCUDA object and copies initial data to the GPU.
NetworkCUDA_Interface::NetworkCUDA_Interface(
    const NetworkConfig& config,
    const std::vector<GPUNeuronState>& neurons,
    const std::vector<GPUSynapse>& synapses)
{
    // Create the CUDA network manager with default CUDA config
    NetworkCUDA::CUDAConfig cuda_config;
    cuda_network_ = std::make_unique<NetworkCUDA>(cuda_config);
    
    // Modify config to match actual neuron/synapse counts
    NetworkConfig adjusted_config = config;
    adjusted_config.num_neurons = neurons.size();
    
    // Initialize the CUDA network with the adjusted configuration
    auto [success, error_msg] = cuda_network_->initialize(adjusted_config);
    if (!success) {
        throw std::runtime_error("Failed to initialize NetworkCUDA: " + error_msg);
    }
    
    // Copy the provided neuron data to GPU
    // NetworkCUDA allocates d_neurons_ during initialize(), so we overwrite it
    if (!neurons.empty()) {
        GPUNeuronState* d_neurons = cuda_network_->getDeviceNeurons();
        if (d_neurons) {
            cudaError_t err = cudaMemcpy(d_neurons, neurons.data(),
                                        neurons.size() * sizeof(GPUNeuronState),
                                        cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                std::cerr << "⚠️  Failed to copy neurons to GPU: " 
                         << cudaGetErrorString(err) << std::endl;
            } else {
                std::cout << "✓ Copied " << neurons.size() << " neurons to GPU" << std::endl;
            }
        }
    }
    
    // Copy the provided synapse data to GPU
    if (!synapses.empty()) {
        GPUSynapse* d_synapses = cuda_network_->getDeviceSynapses();
        if (d_synapses) {
            cudaError_t err = cudaMemcpy(d_synapses, synapses.data(),
                                        synapses.size() * sizeof(GPUSynapse),
                                        cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                std::cerr << "⚠️  Failed to copy synapses to GPU: " 
                         << cudaGetErrorString(err) << std::endl;
            } else {
                std::cout << "✓ Copied " << synapses.size() << " synapses to GPU" << std::endl;
            }
        }
    }
    
    // Synchronize to ensure all data is transferred
    cudaDeviceSynchronize();
}

// --- Destructor ---
NetworkCUDA_Interface::~NetworkCUDA_Interface() = default;

// --- Simulation Step ---
void NetworkCUDA_Interface::step(float current_time, float dt, float reward, const std::vector<float>& inputs) {
    if (cuda_network_) {
        // Use the update method
        cuda_network_->update(dt, reward, 0.0f);
        
        // Process inputs if provided
        if (!inputs.empty()) {
            cuda_network_->processInput(inputs);
        }
    }
}

// --- Get Statistics ---
NetworkStats NetworkCUDA_Interface::get_stats() const {
    NetworkStats stats = {};
    if (cuda_network_) {
        // Get basic stats from the network
        auto memory_stats = cuda_network_->getMemoryStats();
        stats.memory_usage_bytes = memory_stats.allocated_memory_bytes;
    }
    return stats;
}

// --- Get Full Network State ---
void NetworkCUDA_Interface::get_network_state(std::vector<GPUNeuronState>& neurons, std::vector<GPUSynapse>& synapses) {
    if (cuda_network_) {
        // Use existing getNeuronStates method
        neurons = cuda_network_->getNeuronStates();
        
        // For synapses, we need to copy them manually
        // This requires access to device synapses pointer
        // For now, leave as placeholder - this would need a getSynapseStates() method
    }
}