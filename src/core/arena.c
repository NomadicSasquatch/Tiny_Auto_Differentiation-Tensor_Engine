#include "tinyengine.h"

void compute_rowmajor_strides(Tensor* tensor) {
    int64_t accumulate = 1

    for(int i = tensor->ndim - 1; i >= 0; i--) {
        tensor->stride[i] = accumulate;
        accumulate *= tensor->shape[i];
    }
}

Tensor* tensor_new(Arena* arena, int ndim, const int64_t* shape) {
    Tensor* ret_tensor = arena_alloc(arena, sizeof(Tensor), alignof(Tensor));
    ret_tensor->ndim = ndim;
    memcpy(ret_tensor->shape, shape, ndim * sizeof(int64_t));
    compute_rowmajor_strides(ret_tensor);
    
    size_t num_elements = 1;
    for(int i = 0; i < ndim; i++) {
        num_elements *= shape[i];
    }
    
    ret_tensor->data = arena_alloc(arena, num_elements * sizeof(float), alignof(float));
    ret_tensor->grad = NULL;

    return ret_tensor;
}
