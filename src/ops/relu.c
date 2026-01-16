#include "op.h"
#include "graph.h"
#include "tensor.h"
#include "utils.h"

#include <stddef.h>

static void relu_fwd(Node* node) {
    Tensor* A = node->inputs[0]->out;
    Tensor* C = node->out;

    size_t number_elements = total_elems(C);

    for(size_t i = 0; i < number_elements; i++) {
        A->data[i] = A->data[i] > 0.0f? A->data[i] : 0.0f;
    }
}

static void relu_bwd(Node* node) {
    Tensor* A = node->inputs[0]->out;
    Tensor* C = node->out;
    Tensor* gA = A->grad;
    Tensor* gC = C->grad;

    size_t number_elements = total_elems(C);

    for(size_t i = 0; i < number_elements; i++) {
        gA->data[i] += (A->data[i] > 0.0f)? gC->data[i] : 0.0f;
    }
}

static const OpKernel relu_kernel = {
    .optype = OP_RELU,
    .name = "relu",
    .forward = relu_fwd,
    .backward = relu_bwd,
};

__attribute__((constructor))
static void register_add_kernel(void) {
    register_opkernel(&relu_kernel);
}