#pragma once
#include <vector>
#include <random>
#include <cmath>
#include <iostream>

class FeedbackMatrix {
public:
    FeedbackMatrix(int input_dim, int output_dim) 
        : rows_(output_dim), cols_(input_dim) {
        
        weights_.resize(rows_ * cols_);
        
        // Fixed random initialization (The core of Feedback Alignment)
        // We do NOT update these weights. They act as a fixed "mirror".
        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::normal_distribution<float> dist(0.0f, 0.05f);
        
        for(auto& w : weights_) {
            w = dist(gen);
        }
    }

    // Projects error vector from higher dimension (e.g., Output) to lower (e.g., Broca)
    std::vector<float> project(const std::vector<float>& input_error) {
        if (input_error.size() != static_cast<size_t>(cols_)) {
            // Handle size mismatch gracefully or just return empty
            return std::vector<float>(rows_, 0.0f);
        }

        std::vector<float> output_error(rows_, 0.0f);
        
        // Simple matrix-vector multiplication: y = Wx
        for (int r = 0; r < rows_; ++r) {
            float sum = 0.0f;
            for (int c = 0; c < cols_; ++c) {
                sum += weights_[r * cols_ + c] * input_error[c];
            }
            output_error[r] = sum;
        }
        
        return output_error;
    }

private:
    int rows_; // Output dimension (e.g., Broca size)
    int cols_; // Input dimension (e.g., Vocab size)
    std::vector<float> weights_;
};