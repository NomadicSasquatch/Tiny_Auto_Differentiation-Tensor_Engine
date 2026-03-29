#ifndef OPTIM_H
#define OPTIM_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { OPTIM_SGD, OPTIM_ADAM, OPTIM_ADAM_W } Optimiser;

void tensor_zero_grad(Tensor* tensor);
void sgd_step(Tensor* tensor, float lr);


#ifdef __cplusplus
}
#endif

#endif