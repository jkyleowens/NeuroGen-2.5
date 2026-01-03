#include "engine/SparseMatrix.h"
#include <cusparse.h>
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <map>

namespace neurogen {

// CUDA Error Checking Helper
#define CHECK_CUDA(func) { \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA API failed at line " << __LINE__ << " with error: " \
                  << cudaGetErrorString(status) << " (" << status << ")" << std::endl; \
    } \
}

#define CHECK_CUSPARSE(func) { \
    cusparseStatus_t status = (func); \
    if (status != CUSPARSE_STATUS_SUCCESS) { \
        std::cerr << "CUSPARSE API failed at line " << __LINE__ << " with error: " \
                  << status << std::endl; \
    } \
}

struct SparseMatrixImpl {
    cusparseHandle_t handle;
    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecX = nullptr;
    cusparseDnVecDescr_t vecY = nullptr;
    void* dBuffer = nullptr;
    size_t bufferSize = 0;
    bool descriptors_created = false;

    SparseMatrixImpl() {
        CHECK_CUSPARSE(cusparseCreate(&handle));
    }

    ~SparseMatrixImpl() {
        if (dBuffer) CHECK_CUDA(cudaFree(dBuffer));
        if (descriptors_created) {
            if (matA) CHECK_CUSPARSE(cusparseDestroySpMat(matA));
            if (vecX) CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
            if (vecY) CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
        }
        CHECK_CUSPARSE(cusparseDestroy(handle));
    }
};

SparseMatrix::SparseMatrix() : 
    num_rows_(0), num_cols_(0), num_nonzeros_(0), capacity_(0),
    d_values_(nullptr), d_col_indices_(nullptr), d_row_ptrs_(nullptr) {
    
    internal_handle_ = new SparseMatrixImpl();
}

SparseMatrix::~SparseMatrix() {
    cleanup();
    delete static_cast<SparseMatrixImpl*>(internal_handle_);
}

void SparseMatrix::initialize(size_t rows, size_t cols, size_t capacity) {
    num_rows_ = rows;
    num_cols_ = cols;
    capacity_ = capacity;
    
    // Allocate GPU memory for CSR format
    // Row pointers: [rows + 1]
    CHECK_CUDA(cudaMalloc(&d_row_ptrs_, (rows + 1) * sizeof(int)));
    
    // Values and Column Indices: [capacity]
    CHECK_CUDA(cudaMalloc(&d_col_indices_, capacity * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_values_, capacity * sizeof(float)));

    // Initialize row pointers to zero
    CHECK_CUDA(cudaMemset(d_row_ptrs_, 0, (rows + 1) * sizeof(int)));
}

void SparseMatrix::cleanup() {
    if (d_values_) { CHECK_CUDA(cudaFree(d_values_)); d_values_ = nullptr; }
    if (d_col_indices_) { CHECK_CUDA(cudaFree(d_col_indices_)); d_col_indices_ = nullptr; }
    if (d_row_ptrs_) { CHECK_CUDA(cudaFree(d_row_ptrs_)); d_row_ptrs_ = nullptr; }
    num_nonzeros_ = 0;
}

void SparseMatrix::setFromTriplets(const std::vector<int>& rows, 
                                   const std::vector<int>& cols, 
                                   const std::vector<float>& values) {
    if (rows.size() != cols.size() || rows.size() != values.size()) {
        std::cerr << "Error: Triplet vectors must have same size" << std::endl;
        return;
    }

    size_t nnz = rows.size();
    if (nnz > capacity_) {
        std::cerr << "Error: Number of non-zeros exceeds capacity" << std::endl;
        return;
    }
    num_nonzeros_ = nnz;

    // Host-side conversion to CSR
    std::vector<int> h_row_ptrs(num_rows_ + 1, 0);
    std::vector<int> h_col_indices(nnz);
    std::vector<float> h_values(nnz);

    // 1. Count entries per row
    for (int r : rows) {
        if (r >= 0 && r < num_rows_) {
            h_row_ptrs[r + 1]++;
        }
    }

    // 2. Cumulative sum for row pointers
    for (size_t i = 0; i < num_rows_; ++i) {
        h_row_ptrs[i + 1] += h_row_ptrs[i];
    }

    // 3. Fill arrays (assuming input triplets are sorted by row, then col is ideal, 
    //    but we'll do a basic fill. For correct CSR, cols within a row should be sorted,
    //    but for simple SpMV often not strictly required if we just insert in order).
    //    Better approach: Use a vector of vectors or sort the triplets first.
    
    // Simple sort-based approach
    struct Triplet { int r, c; float v; };
    std::vector<Triplet> triplets(nnz);
    for(size_t i=0; i<nnz; ++i) {
        triplets[i] = {rows[i], cols[i], values[i]};
    }
    
    std::sort(triplets.begin(), triplets.end(), [](const Triplet& a, const Triplet& b) {
        if (a.r != b.r) return a.r < b.r;
        return a.c < b.c;
    });

    // Re-calculate row pointers based on sorted data to be safe
    std::fill(h_row_ptrs.begin(), h_row_ptrs.end(), 0);
    for (const auto& t : triplets) {
        h_row_ptrs[t.r + 1]++;
    }
    for (size_t i = 0; i < num_rows_; ++i) {
        h_row_ptrs[i + 1] += h_row_ptrs[i];
    }

    // Fill cols and values
    for (size_t i = 0; i < nnz; ++i) {
        h_col_indices[i] = triplets[i].c;
        h_values[i] = triplets[i].v;
    }

    // Copy to GPU
    CHECK_CUDA(cudaMemcpy(d_row_ptrs_, h_row_ptrs.data(), (num_rows_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_indices_, h_col_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values_, h_values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));

    // Update cuSPARSE descriptors
    SparseMatrixImpl* impl = static_cast<SparseMatrixImpl*>(internal_handle_);
    
    if (impl->matA) CHECK_CUSPARSE(cusparseDestroySpMat(impl->matA));
    
    CHECK_CUSPARSE(cusparseCreateCsr(&impl->matA, 
                                   num_rows_, num_cols_, num_nonzeros_,
                                   d_row_ptrs_, d_col_indices_, d_values_,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
}

void SparseMatrix::vectorMultiply(const float* d_x, float* d_y, float alpha, float beta) {
    if (num_nonzeros_ == 0) return;

    SparseMatrixImpl* impl = static_cast<SparseMatrixImpl*>(internal_handle_);

    // Create vector descriptors if needed (pointers act as handles)
    // Note: In a real engine, we might cache these better, but we need to support changing pointers
    if (impl->vecX) CHECK_CUSPARSE(cusparseDestroyDnVec(impl->vecX));
    if (impl->vecY) CHECK_CUSPARSE(cusparseDestroyDnVec(impl->vecY));

    // Const cast needed because API takes void* but treats as const for input
    CHECK_CUSPARSE(cusparseCreateDnVec(&impl->vecX, num_cols_, (void*)d_x, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&impl->vecY, num_rows_, (void*)d_y, CUDA_R_32F));

    // Buffer management
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        impl->handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, impl->matA, impl->vecX, &beta, impl->vecY, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));

    if (bufferSize > impl->bufferSize) {
        if (impl->dBuffer) CHECK_CUDA(cudaFree(impl->dBuffer));
        CHECK_CUDA(cudaMalloc(&impl->dBuffer, bufferSize));
        impl->bufferSize = bufferSize;
    }

    // Execute SpMV
    CHECK_CUSPARSE(cusparseSpMV(
        impl->handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, impl->matA, impl->vecX, &beta, impl->vecY, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, impl->dBuffer));
        
    // Clean up ephemeral descriptors
    CHECK_CUSPARSE(cusparseDestroyDnVec(impl->vecX)); impl->vecX = nullptr;
    CHECK_CUSPARSE(cusparseDestroyDnVec(impl->vecY)); impl->vecY = nullptr;
}

} // namespace neurogen

