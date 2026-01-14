#include "op.h"
#include "tensor.h"

#include <stddef.h>

static void matmul_fwd(Node* node) {
    Tensor* A =  node->inputs[0]->out;
    Tensor* B = node->inputs[1]->out;
    Tensor* C = node->out;

    size_t number_elements = total_elems(C);
}

static const OpKernel add_kernel = {
    .op_type = OP_MATMUL,
    .name = "mat_mul",
    .forward = matmul_fwd,
    .backward = matmul_bwd,
};