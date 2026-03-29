#include "op.h"

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
    Tensor* gA = node->inputs[0]->out->grad;
    Tensor* gB = node->inputs[1]->out->grad;
    Tensor* gC = node->out->grad;
    size_t number_elements = total_elems(gC);
    
    for(size_t i = 0; i < number_elements; i++) {
        gA->data[i] += gC->data[i];
        gB->data[i] += gC->data[i];
    }
}

static const OpKernel add_kernel = {
    .optype = OP_ADD,
    .name = "add",
    .forward = add_fwd,
    .backward = add_bwd,
};

// Run before main is called
__attribute__((constructor))
static void register_add_kernel(void) {
    register_opkernel(&add_kernel);
}

#ifdef ADD_SELFTEST_MAIN

int main(void) {
    const int64_t dim_a[2] = {2, 3};
    float fill_a[2] = {2.0, 7.0};
    float fill_b[2] = {3.0, 8.0};

    testOp(OP_ADD, dim_a, dim_a, dim_a, fill_a, fill_b, 5.0, NULL);
    return 0;
}
#endif