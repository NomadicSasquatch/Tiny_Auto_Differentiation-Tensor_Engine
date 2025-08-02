#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/mman.h>

// use C linkage for any of the libraries that are in cpp
#ifdef __cplusplus
extern "C" {
#endif  

// defining logic to break out of program and debug
#define fatal(format, ...)                       \
    do {                                         \
        fprintf(stderr, "Fatal: " format "\n",   \
                ##__VA_ARGS__);                  \
        exit(EXIT_FAILURE);                      \
    } while(0)

// tensor struct, grad for easier (lazier) backpropagation
// data is all stored within our arena
typedef struct {
    float* data;
    int64_t shape[6];
    int64_t stride[6];
    int ndim;
    struct Tensor* grad;
} Tensor;

// NTS: const so we compiler would yell at us if we accidentally change the input tensor
size_t total_elems(const Tensor* tensor);

#ifdef __cplusplus
}
#endif
#endif