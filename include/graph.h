#ifndef GRAPH_H
#define GRAPH_H

#include "op.h"
#include "tensor.h"
#include "arena.h"

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
// NTS: The node struct allows for multiple inputs, but the code for the operators tentatively are hard coded to 2 inputs. If more inputs are permitted I think it would only be in the context of fused kernels, but have to write out the backwards as well and change the main loops for the basic forward and backward as well
typedef struct {
    Op operation;
    Tensor* out;
    struct Node** inputs;
    int n_input;

    // Meta data for possible toposort
    int topo_index;
    atomic_int pending_parents;
    // Add children?
} Node;

typedef struct {
    Arena* arena;
    Node** nodes;
    size_t size;
    size_t capacity;
} Graph;

void graph_init(Graph *g, Arena* arena);
void graph_free(Graph* g);
// Leaf or input node, to wrap an existing tensor as the input node
Node* graph_add_input(Graph* g, Tensor* t);
Tensor* add_node(Graph *g, Op op, int n_in, Node **inputs);
void topo_sort(Graph *g, Node ***out_order, size_t *out_n);
void forward(Graph *g, Node **order, size_t n);
void backward(Graph *g, Node **order, size_t n, Tensor *loss);

#ifdef __cplusplus
}
#endif
#endif