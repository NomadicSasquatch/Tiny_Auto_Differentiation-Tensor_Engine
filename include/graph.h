#ifndef GRAPH_H
#define GRAPH_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
#include <pthread.h>
#include <sys/mman.h>
#include <errno.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>
#include <stdarg.h>

// use C linkage for any of the libraries that are in cpp
#ifdef __cplusplus
extern "C" {
#endif  

// atomic int to prevent race conditions
// we will be using toposort - topo_index for index in topo order, parents is the atomic counter for Kahn's algo
typedef struct {
    Op operation;
    Tensor* out;
    struct Node* inputs[3];
    int n_input;
    int topo_index;
    atomic_int pending_parents;
    // add children?
} Node;

typedef struct {
    Node** nodes;
    size_t size;
    size_t capacity;
} Graph;

void graph_init(Graph *g);
Tensor* add_node(Graph *g, Op op, int n_in, Node **inputs);
void topo_sort(Graph *g, Node ***out_order, size_t *out_n);
void forward(Graph *g, Node **order, size_t n);
void backward(Graph *g, Node **order, size_t n, Tensor *loss);

#ifdef __cplusplus
}
#endif
#endif