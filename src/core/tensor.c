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

    // Linear memory layout, last dim always increments by 1
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

Tensor* tensor_new(Arena* arena, int ndim, const int64_t* shape) {
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
        memcpy(tensor->shape, shape, (size_t) ndim * sizeof(int64_t));
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
        return NULL;
    }

    Tensor* new_tensor = tensor_new(arena, like->ndim, like->shape);
    tensor_fill(new_tensor, 0.0f);

    return new_tensor;
}

void print_tensor_recursive(const Tensor* t, int dim, int64_t offset) {
    if (dim == t->ndim) {
        printf("%g", t->data[offset]);
        return;
    }

    printf("[");
    for (int64_t i = 0; i < t->shape[dim]; i++) {
        if (i > 0) {
            if (dim == t->ndim - 1) {
                printf(", ");
            } else {
                printf(",\n");
                for (int j = 0; j < dim + 1; j++) {
                    printf(" ");
                }
            }
        }
        print_tensor_recursive(t, dim + 1, offset + i * t->stride[dim]);
    }
    printf("]");
}

void print_tensor(const Tensor* t) {
    if (!t) {
        printf("Tensor(NULL)\n");
        return;
    }

    printf("Tensor(\n");

    printf("  ndim=%d,\n", t->ndim);

    printf("  shape=[");
    for (int i = 0; i < t->ndim; i++) {
        printf("%lld", (long long)t->shape[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf("],\n");

    printf("  stride=[");
    for (int i = 0; i < t->ndim; i++) {
        printf("%lld", (long long)t->stride[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf("],\n");

    printf("  data=");
    if (!t->data) {
        printf("NULL\n");
    } else if (t->ndim == 0) {
        printf("%g\n", t->data[0]);
    } else {
        print_tensor_recursive(t, 0, 0);
        printf("\n");
    }

    printf(")\n");
}

#ifdef TENSOR_SELFTEST_MAIN

int main(void) {
    Arena a;
    Tensor* t;
    arena_init(&a, 4096);
    const int64_t t_shape[2] = {2, 3};

    t = tensor_new(&a, 2, t_shape);
    print_tensor(t);

    tensor_fill(t, 3.0);
    print_tensor(t);

    Tensor* t2 = tensor_zeroes_like(&a, t);
    print_tensor(t2);


    arena_free(&a);
    return 0;
}

#endif