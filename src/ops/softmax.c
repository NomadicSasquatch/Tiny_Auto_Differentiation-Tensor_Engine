#include "op.h"
#include "graph.h"
#include "tensor.h"
#include "utils.h"

#include <stddef.h>
#include <math.h>
#include <stdio.h>

// Take softmax as softmax per row, only for up till 2D (taking it as softmax per batch)
static void softmax_fwd(Node* node) {
    Tensor* A = node->inputs[0]->out;
    Tensor* C = node->out;

    int64_t batch = 1;
    int64_t dimension = 0;
    int64_t stride_b = 0;
    int64_t stride_d = 0;

    if(A->ndim == 1) {
        batch = 1;
        dimension = A->shape[0];
        stride_b = 0;
        stride_d = 1;
    }
    else if(A->ndim == 2) {
        batch = A->shape[0];
        dimension = A->shape[1];
        stride_b = A->stride[0];
        stride_d = A->stride[1];
    }
    else {
        fatal("softmax_fwd cannot run: input tensor has too many dimensions of %d, keep to <= 2", A->ndim);
    }

    int64_t c_stride_b = (C->ndim == 1)? 0 : C->stride[0];
    int64_t c_stride_d = (C->ndim == 1)? C->stride[0] : C->stride[1];

    for(int64_t b = 0; b < batch; b++) {
        // max for numerical stability
        float max_ = -INFINITY;
        for(int64_t d = 0; d < dimension; d++) {
            size_t a_idx = (size_t)(b * stride_b + d * stride_d);
            float val = A->data[a_idx];
            max_ = (max_ > val) ? max_ : val;
        }

        float sum = 0.0f;
        for(int64_t d = 0; d < dimension; d++) {
            size_t a_idx = (size_t)(b * stride_b + d * stride_d);
            size_t c_idx = (size_t)(b * c_stride_b + d * c_stride_d);
            float exp_value = expf(A->data[a_idx] - max_);
            C->data[c_idx] = exp_value;
            sum += exp_value;
        }

        for(int64_t d = 0; d < dimension; d++) {
            size_t c_idx = (size_t)(b * c_stride_b + d * c_stride_d);
            C->data[c_idx] /= sum;
        }
    }
}

// Given Y = softmax(X), the adjoint should be dX_i = Y_i * (dY_i - dot(dY, Y))
static void softmax_bwd(Node* node) {
    Tensor* A = node->inputs[0]->out;
    Tensor* C = node->out;
    Tensor* gA = A->grad;
    Tensor* gC = C->grad;

    int64_t batch = 1;
    int64_t dimension = 0;

    if(A->ndim == 1) {
        batch = 1;
        dimension = A->shape[0];
    }
    else if(A->ndim == 2) {
        batch = A->shape[0];
        dimension = A->shape[1];
    }
    else {
        fatal("softmax_bwd cannot run: input tensor has too many dimensions of %d, keep to <= 2", A->ndim);
    }

    int64_t c_stride_b = (C->ndim == 1)? 0 : C->stride[0];
    int64_t c_stride_d = (C->ndim == 1)? C->stride[0] : C->stride[1];

    // dX_i = Y_i * (dY_i - dot(dY, Y))
    for(int64_t b = 0; b < batch; b++) {
        float dot = 0.0f;

        for(int64_t d = 0; d < dimension; d++) {
            size_t idx = (size_t)(b * c_stride_b + d * c_stride_d);
            dot += gC->data[idx] * C->data[idx];
        }

        for(int64_t d = 0; d < dimension; d++) {
            size_t idx = (size_t)(b * c_stride_b + d * c_stride_d);
            gA->data[idx] += C->data[idx] * (gC->data[idx] - dot);
        }
    }
}

static const OpKernel softmax_kernel = {
    .optype = OP_SOFTMAX,
    .name = "softmax",
    .forward = softmax_fwd,
    .backward = softmax_bwd,
};

__attribute__((constructor))
static void register_add_kernel(void) {
    register_opkernel(&softmax_kernel);
}

#ifdef SOFTMAX_SELFTEST_MAIN

int main(void) {
    const int64_t dim_a[2] = {2, 3};
    float unary_out[6] = {0.333333, 0.333333, 0.333333, 0.333333, 0.333333, 0.333333}; 

    testOp(OP_SOFTMAX, dim_a, dim_a, dim_a, 2.0, 3.0, 6.0, unary_out);
    return 0;
}
#endif