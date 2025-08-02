#include "op.h"
#include "tensor.h"

#include <stddef.h>

static void add_fwd(Node* node) {
    Tensor* A = node->inputs[0]->out;
    Tensor* B = node->inputs[1]->out;
    Tensor* C = node->out;
    size_t number_elements = total_elems(C);

    for(size_t i = 0; i < number_elements; i++) {
        C->data[i] = A->data[i] + B->data[i];
    }
}

static void add_bwd(Node* node) {
    Tensor* tensor = node->out->grad;
    size_t number_elements = total_elems(tensor);
    
    memcpy(node->inputs[0]->out->grad->data, tensor->data, sizeof(float) * number_elements);
    memcpy(node->inputs[1]->out->grad->data, tensor->data, sizeof(float) * number_elements);
}

static const OpKernel add_kernel = {
    .op_type = OP_ADD,
    .name    = "add",
    .fwd     = add_fwd,
    .bwd     = add_bwd,
};

// Run before main is called
__attribute__((constructor))
static void register_add_kernel(void) {
    register_opkernel(&add_kernel);
}
