#include "op.h"
#include "graph.h"
#include "tensor.h"
#include "utils.h"

#include <stddef.h>

static void sub_fwd(Node* node) {
    Tensor* A = node->inputs[0]->out;
    Tensor* B = node->inputs[1]->out;
    Tensor* C = node->out;
    size_t number_elements = total_elems(C);

    for(size_t i = 0; i < number_elements; i++) {
        C->data[i] = A->data[i] - B->data[i];
    }
}

static void sub_bwd(Node* node) {
    Tensor* gA = node->inputs[0]->out->grad;
    Tensor* gB = node->inputs[1]->out->grad;
    Tensor* gC = node->out->grad;
    size_t number_elements = total_elems(gC);

    for(size_t i = 0; i < number_elements; i++) {
        gA->data[i] += gC->data[i];
        gB->data[i] -= gC->data[i];
    }
}

static const OpKernel sub_kernel = {
    .optype = OP_SUB,
    .name = "sub",
    .forward = sub_fwd,
    .backward = sub_bwd,
};

__attribute__((constructor))
static void register_sub_kernel(void) {
    register_opkernel(&sub_kernel);
}

#ifdef SUB_SELFTEST_MAIN

int main(void) {
    const int64_t dim_a[2] = {2, 3};

    testOp(OP_SUB, dim_a, dim_a, dim_a, 3.0, 2.0, 1.0);
    return 0;
}
#endif