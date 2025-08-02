#include "tensor.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/mman.h>
#include <assert.h>

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