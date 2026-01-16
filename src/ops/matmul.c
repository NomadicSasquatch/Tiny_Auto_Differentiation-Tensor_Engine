#include "op.h"
#include "graph.h"
#include "tensor.h"
#include "utils.h"

#include <stddef.h>

// Here we just take the matmul to be C = A X B
// Helper functions for indexing using the strides
static inline float at(const Tensor* tensor, int64_t i, int64_t j) {
    return tensor->data[(size_t) (i * tensor->stride[0] + j * tensor->stride[1])];
}

static inline void add(const Tensor* tensor, int64_t i, int64_t j, float value) {
    tensor->data[(size_t) (i * tensor->stride[0] + j * tensor->stride[1])] += value;
}

static inline void set(const Tensor* tensor, int64_t i, int64_t j, float value) {
    tensor->data[(size_t) (i * tensor->stride[0] + j * tensor->stride[1])] = value;
}

static void matmul_fwd(Node* node) {
    // Take shape as [n,m]
    Tensor* A =  node->inputs[0]->out;
    // Take shape as [m,k]
    Tensor* B = node->inputs[1]->out;
    // Take shape as [n,k]
    Tensor* C = node->out;

    // Dimension checking is done before this function is called in graph.c
    int64_t n = A->shape[0];
    int64_t m = A->shape[1];
    int64_t k = B->shape[1];

    for(int64_t i = 0; i < n; i++) {
        for(int64_t j = 0; j < k; j++) {
            float sum = 0.0f;

            for(int64_t l = 0; l < m; l++) {
                sum += at(A, i, l) * at(B, l, j);
            }
            
            set(C, i, j, sum);
        }
    }
}

static void matmul_bwd(Node* node) {
    // Take shape as [n,m]
    Tensor* A =  node->inputs[0]->out;
    // Take shape as [m,k]
    Tensor* B = node->inputs[1]->out;
    // Take shape as [n,k]
    Tensor* C = node->out;

    Tensor* gA = A->grad;
    Tensor* gB = B->grad;
    Tensor* gC = C->grad;

    // Dimension checking is done before this function is called in graph.c
    int64_t n = A->shape[0];
    int64_t m = A->shape[1];
    int64_t k = B->shape[1];

    // Partial adjoint for given A is dA = dC @ B^T, we accumulate this
    for(int64_t i = 0; i < n; i++) {
        for(int64_t j = 0; j < m; j++) {
            float sum = 0.0f;

            for(int64_t l = 0; l < k; l++) {
                sum += at(gC, i, l) * at(B, j, l);
            }

            add(gA, i, j, sum);
        }
    }

    // Partial adjoint for given B is dB = A^T @ dC, we accumulate this (Double check dim iteration)
    for(int64_t i = 0; i < m; i++) {
        for(int64_t j = 0; j < k; j++) {
            float sum = 0.0f;

            for(int64_t l = 0; l < n; l++) {
                sum += at(A, l, i) + at(gC, l, j);
            }

            add(gB, i, j, sum);
        }
    }
}

static const OpKernel mat_mul_kernel = {
    .optype = OP_MATMUL,
    .name = "mat_mul",
    .forward = matmul_fwd,
    .backward = matmul_bwd,
};

__attribute__((constructor))
static void register_mul_kernel(void) {
    register_opkernel(&mat_mul_kernel);
}