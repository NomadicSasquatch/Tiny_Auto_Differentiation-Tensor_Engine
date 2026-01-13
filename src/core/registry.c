#include "op.h"
#define MAX_OPS 20

static const OpKernel* registry[MAX_OPS];
static int registry_idx = 0;

void register_opkernel(const OpKernel* kernel) {
    if(kernel->optype >= MAX_OPS) {
        fatal("Registry is full!");
    }
    registry[kernel->optype] = kernel;
}

const OpKernel* get_opkernel(Op optype) {
    return registry[optype];
}