#include "op.h"
#include "tensor.h"

#include <stddef.h>

static void mul_fwd(Node* node) {
    Tensor* A = node->inputs[0]->out;
    Tensor* B = node->inputs[1]->out;
    Tensor* C = node->out;
    size_t number_elements = total_elems(C);

    for(size_t i = 0; i < number_elements; i++) {
        C->data[i] = A->data[i] * B->data[i];
    }
}

static void mul_bwd(Node* node) {
    Tensor* A  = node->inputs[0]->out;
    Tensor* B  = node->inputs[1]->out;
    Tensor* C  = node->out;
    Tensor* gC = C->grad;
    size_t number_elements = total_elems(C);

    for(size_t i = 0; i < number_elements; i++) {
        A->grad->data[i] += B->data[i] * gC->data[i];
        B->grad->data[i] += A->data[i] * gC->data[i];
    }
}

static const OpKernel mul_kernel = {
    .op_type = OP_MUL,
    .name = "mul",
    .forward = mul_fwd,
    .backward = mul_bwd,
};

__attribute__((constructor))
static void register_mul_kernel(void) {
    register_opkernel(&mul_kernel);
}