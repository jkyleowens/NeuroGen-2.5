#include <iostream>
#include <memory>
#include <string>
#include <chrono>
#include <thread>
#include "modules/BrainOrchestrator.h"

// Phase 2 Main Loop with "High-Performance" mode
int main(int argc, char** argv) {
    // Configuration
    // To "maximize model's capabilities" while "accounting for compute limits":
    // We run as fast as possible (unlocked frame rate) but monitor performance.
    // If we drop below target Hz, we might need to prune or scale down (Phase 3).
    // For Phase 2, we focus on pure throughput measurement.

    std::cout << "🧠 NeuroGen 2.0: High-Performance Emulation Environment" << std::endl;
    
    BrainOrchestrator::Config brain_config;
    brain_config.gpu_device_id = 0;
    brain_config.time_step_ms = 1.0f;
    brain_config.enable_parallel_execution = true;
    brain_config.enable_consolidation = false; // Disable complex features for raw speed benchmark
    brain_config.processing_mode = BrainOrchestrator::ProcessingMode::PIPELINED; // Enable pipelined mode
    brain_config.max_pipeline_depth = 8; // Allow up to 8 tokens in pipeline before forcing output

    auto brain = std::make_unique<BrainOrchestrator>(brain_config);
    
    std::cout << "🚀 Initializing Neural Modules..." << std::endl;
    brain->initializeModules();
    brain->createConnectome();
    
    std::cout << "⚡ Starting Main Simulation Loop (Compute Phase 2)" << std::endl;
    
    // Input vector for stimulation (random noise)
    std::vector<float> input(256, 1.0f); // Thalamus input size (Direct Character Input)
    
    // Performance Tracking
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_report_time = start_time;
    long long total_steps = 0;
    
    // Simulation Loop
    while (true) {
        // 1. Cognitive Step (Full Brain Update)
        brain->cognitiveStep(input);
        
        total_steps++;
        
        // 2. Performance Reporting (Every 1000 steps)
        if (total_steps % 1000 == 0) {
            auto current_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = current_time - last_report_time;
            double tps = 1000.0 / elapsed.count(); // Steps per second
            
            std::cout << "Step " << total_steps << ": " 
                      << tps << " Hz (Simulated Time: " << total_steps * brain_config.time_step_ms << " ms)" 
                      << std::endl;
                      
            last_report_time = current_time;
            
            // Compute Limits check (simulated)
            // If TPS drops below threshold, we might warn user
            if (tps < 100.0) {
                std::cout << "⚠️ Warning: Simulation speed below real-time target (1000 Hz). "
                          << "Current: " << tps << " Hz" << std::endl;
            }
        }
        
        // Optional: Break condition
        if (total_steps >= 10000) break; // Run for 10k steps then exit for this test
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = end_time - start_time;
    
    std::cout << "✅ Simulation Complete." << std::endl;
    std::cout << "Total Time: " << total_elapsed.count() << " s" << std::endl;
    std::cout << "Average Speed: " << total_steps / total_elapsed.count() << " Hz" << std::endl;

    return 0;
}
