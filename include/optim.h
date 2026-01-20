#ifndef OPTIM_H
#define OPTIM_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

void tensor_zero_grad(Tensor* tensor);
void sgd_step(Tensor* tensor, float lr);


#ifdef __cplusplus
}
#endif

#endif