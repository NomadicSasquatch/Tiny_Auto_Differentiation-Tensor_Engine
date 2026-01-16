#include "tensor.h"
#include "utils.h"
#include "arena.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/mman.h>
#include <assert.h>
#include <stdalign.h>

void compute_rowmajor_strides(Tensor* tensor) {
    int64_t accumulate = 1;

    for(int i = tensor->ndim - 1; i >= 0; i--) {
        tensor->stride[i] = accumulate;
        accumulate *= tensor->shape[i];
    }
    // Hardcoded 6 since the dimension limit of tensors as declared in the header file is 6.
    for(int i = tensor->ndim; i < 6; i++) {
        tensor->stride[i] = 0;
        tensor->shape[i] = 0;
    }
}

size_t total_elems(const Tensor* tensor) {
    assert(tensor != NULL);
    assert(tensor->ndim >= 0 && tensor->ndim <= 6);

    if(tensor->ndim == 0) {
        return 1;
    }

    size_t prod = 1;
    
    for(int i = 0; i < tensor->ndim; i++) {
        int64_t dim = tensor->shape[i];
        assert(dim >= 0);
        prod *= (size_t)dim;
    }

    return prod;
}

Tensor* tensor_new(Arena* arena, int ndim, const uint64_t* shape) {
    if(!arena) {
        fatal("tensor_new cannot run: arena is NULL");
    }
    if(ndim < 0 || ndim > 6) {
        fatal("tensor_new cannot run: tensor ndim is out of range %d < 0 || %d > 6", ndim, ndim);
    }

    Tensor* tensor = (Tensor*) arena_alloc(arena, sizeof(Tensor), alignof(Tensor));
    memset(tensor, 0, sizeof(Tensor));

    tensor->ndim = ndim;

    if(ndim > 0) {
        memcpy(tensor->shape, shape, (size_t)ndim * sizeof(uint64_t));
    }

    compute_rowmajor_strides(tensor);

    size_t n = total_elems(tensor);
    tensor->data = (float*) arena_alloc(arena, n * sizeof(float), alignof(float));
    tensor->grad = NULL;

    return tensor;
}

void tensor_fill(Tensor* tensor, float value) {
    if(!tensor) {
        return;
    }

    size_t n = total_elems(tensor);
    
    for(size_t i = 0; i < n; i++) {
        tensor->data[i] = value;
    }
}

Tensor* tensor_zeroes_like(Arena* arena, const Tensor* like) {
    if(!like) {
        return;
    }

    Tensor* new_tensor = tensor_new(arena, like->ndim, like->shape);
    tensor_fill(new_tensor, 0.0f);

    return new_tensor;
}