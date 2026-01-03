#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <memory>

namespace neurogen {

/**
 * @brief Sparse Matrix implementation (CSR/BSR format) for synaptic weights.
 * 
 * Handles storage and computation of synaptic inputs using Sparse Matrix-Vector Multiplication (SpMV).
 * Replaces object-oriented synaptic structures.
 */
class SparseMatrix {
public:
    SparseMatrix();
    ~SparseMatrix();

    /**
     * @brief Initialize the sparse matrix structure
     * @param rows Number of rows (post-synaptic neurons)
     * @param cols Number of columns (pre-synaptic neurons)
     * @param capacity Initial capacity for non-zero elements (synapses)
     */
    void initialize(size_t rows, size_t cols, size_t capacity);

    /**
     * @brief Clean up GPU resources
     */
    void cleanup();

    /**
     * @brief Perform Sparse Matrix-Vector Multiplication (SpMV)
     * y = alpha * A * x + beta * y
     * 
     * @param d_x Input vector (pre-synaptic spikes/activations) on GPU
     * @param d_y Output vector (post-synaptic currents) on GPU
     * @param alpha Scaling factor for A*x
     * @param beta Scaling factor for y (accumulation)
     */
    void vectorMultiply(const float* d_x, float* d_y, float alpha = 1.0f, float beta = 0.0f);

    /**
     * @brief Populate matrix from host-side triplet data (row, col, value)
     * Useful for initialization.
     */
    void setFromTriplets(const std::vector<int>& rows, 
                         const std::vector<int>& cols, 
                         const std::vector<float>& values);

    // === Accessors ===
    float* getValues() const { return d_values_; }
    int* getColIndices() const { return d_col_indices_; }
    int* getRowPtrs() const { return d_row_ptrs_; }
    
    size_t getNumRows() const { return num_rows_; }
    size_t getNumCols() const { return num_cols_; }
    size_t getNumNonZeros() const { return num_nonzeros_; }

private:
    size_t num_rows_;
    size_t num_cols_;
    size_t num_nonzeros_;
    size_t capacity_;

    // CSR Data Arrays (Device Pointers)
    float* d_values_;       // [nnz] Weights
    int* d_col_indices_;    // [nnz] Column indices
    int* d_row_ptrs_;       // [rows + 1] Row pointers

    // Internal implementation details (PIMPL or void* to avoid header dependencies)
    void* internal_handle_; 
};

} // namespace neurogen

