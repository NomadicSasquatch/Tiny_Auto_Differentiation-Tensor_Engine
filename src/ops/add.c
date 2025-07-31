#include "op.h"
#include "tensor.h"

static void add_fwd(Node* n) {
    Tensor* A = n->inputs[0]->out;
    Tensor* B = n->inputs[1]->out;
    Tensor* C = n->out;

    for(size_t i = 0, ne = total_elems(C); i < ne; ++i) {
        C->data[i] = A->data[i] + B->data[i];
    }
}

static void add_bwd(Node* n) {
    Tensor* g = n->out->grad;
    memcpy(n->inputs[0]->out->grad->data, g->data,
           sizeof(float) * total_elems(g));
    memcpy(n->inputs[1]->out->grad->data, g->data,
           sizeof(float) * total_elems(g));
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
