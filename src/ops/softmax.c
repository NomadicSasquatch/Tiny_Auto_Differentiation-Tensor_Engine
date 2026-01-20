#include "op.h"
#include "graph.h"
#include "tensor.h"
#include "utils.h"

#include <stddef.h>
#include <math.h>

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

    // Get strides for the output so we know how to update the values
    int64_t c_stride_b = (C->ndim == 1)? 0 : C->stride[0];
    int64_t c_stride_d = (C->ndim == 1)? C->stride[0]: C->stride[1];

    for(int64_t b = 0; b < batch; b++) {
        // Getting max for the numerically stable softmax
        float max_ = -INFINITY;
        for(int64_t d = 0; d < dimension; d++) {
            size_t idx = (size_t) (b * A->shape[0] + d * A->shape[1]);
            float tmp_feature = A->data[idx];
            max_ = (max_ > tmp_feature)? max_ : tmp_feature;
        }

        float sum = 0.0f;
        for(int64_t d = 0; d < dimension; d++) {
            size_t idx = (size_t) (b * A->shape[0] + d * A->shape[1]);
            float tmp_feature = A->data[idx];
            float exp_value = expf((tmp_feature) - max_);

            sum += exp_value;
            A->data[idx] = exp_value;
        }

        for(int64_t d = 0; d < dimension; d++) {
            size_t idx = (size_t) (b * A->shape[0] + d * A->shape[1]);
            float tmp_feature = A->data[idx];

            A->data[idx] /= sum;
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
    int64_t c_stride_d = (C->ndim == 1)? C->stride[0]: C->stride[1];

    // dX_i = Y_i * (dY_i - dot(dY, Y))
    for(int64_t b = 0; b < batch; b++) {
        float dot = 0.0f;
        
        for(int64_t d = 0; d < dimension; d++) {
            size_t idx = (size_t) (b * c_stride_b + d * c_stride_d);
            dot += gC->data[idx] * C->data[idx];
        }

        for(int64_t d = 0; d < dimension; d++) {
            size_t idx = (size_t) (b * c_stride_b + d * c_stride_d);
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