#ifndef NETWORK_CUDA_INTERFACE_H
#define NETWORK_CUDA_INTERFACE_H

#include <vector>
#include <string>
#include <memory>
#include "NetworkConfig.h"
#include "NetworkStats.h"
#include "GPUNeuralStructures.h"
#include "NetworkCUDA.cuh"

/**
 * @class NetworkCUDA_Interface
 * @brief A user-facing interface to control the CUDA-accelerated neural network.
 *
 * This class wraps the NetworkCUDA object, providing a simplified and stable API
 * for the main application to use, abstracting away the underlying GPU management.
 */
class NetworkCUDA_Interface {
public:
    /**
     * @brief Constructs the CUDA network interface.
     * @param config The network configuration struct.
     * @param neurons A vector of host-side neuron states for initialization.
     * @param synapses A vector of host-side synapse states for initialization.
     */
    NetworkCUDA_Interface(
        const NetworkConfig& config,
        const std::vector<GPUNeuronState>& neurons,
        const std::vector<GPUSynapse>& synapses
    );

    ~NetworkCUDA_Interface();

    /**
     * @brief Runs the simulation for a single time step.
     * @param current_time The current simulation time in milliseconds.
     * @param dt The simulation time step in milliseconds.
     * @param reward The global reward signal for this step.
     * @param inputs A vector of external input currents to be applied.
     */
    void step(float current_time, float dt, float reward, const std::vector<float>& inputs);

    /**
     * @brief Retrieves the current statistics from the network.
     * @return A NetworkStats struct containing simulation metrics.
     */
    NetworkStats get_stats() const;

    /**
     * @brief Copies the current state of the network from the GPU back to the host.
     * @param neurons A host-side vector to be filled with neuron states.
     * @param synapses A host-side vector to be filled with synapse states.
     */
    void get_network_state(std::vector<GPUNeuronState>& neurons, std::vector<GPUSynapse>& synapses);

private:
    // Using a unique_ptr to manage the lifetime of the CUDA network object.
    std::unique_ptr<NetworkCUDA> cuda_network_;
};

#endif // NETWORK_CUDA_INTERFACE_H