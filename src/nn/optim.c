#include "optim.h"
#include "tensor.h"

void tensor_zero_grad(Tensor* tensor) {
    if(!tensor || !tensor->grad) {
        printf("tensor_zero_grad: tensor or gradient is NULL");
        return NULL;
    }

    tensor_fill(tensor->grad, 0.0f);
}

void sgd_step(Tensor* tensor, float lr) {
    if(!tensor || !tensor->grad) {
        printf("sgd_step: tensor or gradient is NULL");
        return NULL;
    }

    size_t number_elements = total_elems(tensor);

    for(size_t i = 0; i < number_elements; i++) {
        tensor->data[i] -= lr * tensor->grad->data[i];
    }
}

// TODO: no adam yet, did not add momentum or velocity fields into tensor struct