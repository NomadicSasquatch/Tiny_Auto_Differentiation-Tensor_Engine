#ifndef GRAPH_H
#define GRAPH_H

#include "op.h"
#include "tensor.h"
#include "arena.h"
#include "utils.h"

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
#include <stdalign.h>

// use C linkage for any of the libraries that are in cpp
#ifdef __cplusplus
extern "C" {
#endif  

typedef struct Tensor Tensor;
typedef struct Arena Arena;

// Singly linked list for children of a node, so we can perform toposort in O(V + E)
typedef struct NodeUse {
    struct Node* user;
    struct NodeUse* next;
} NodeUse;

// atomic int to prevent race conditions
// NTS: The node struct allows for multiple inputs, but the code for the operators tentatively are hard coded to 2 inputs. If more inputs are permitted I think it would only be in the context of fused kernels, but have to write out the backwards as well and change the main loops for the basic forward and backward as well
typedef struct Node {
    Op operation;
    Tensor* out;
    struct Node** inputs;
    int n_input;

    // Meta data for possible toposort
    int topo_index;
    // Children for the curr node
    NodeUse* users;
} Node;

typedef struct {
    Arena* arena;
    Node** nodes;
    size_t size;
    size_t capacity;
} Graph;

void graph_init(Graph* graph, Arena* arena);
void graph_free(Graph* g);
// Leaf or input node, to wrap an existing tensor as the input node
Node* graph_add_input(Graph* g, Tensor* t);
Node* add_node(Graph* graph, Op op, int n_in, Node **inputs);
void graph_ensure_grad(Graph* graph, Tensor* tensor);
void topological_sort(Graph* graph, Node*** output_order, size_t* total_outputs);
void graph_forward_pass(Node* const* order, size_t order_size);
void graph_backward_pass(Graph* graph, Node* const* order, size_t order_size, Tensor* loss);
// Node* graph_optimiser_pass(Graph* graph, Node** order, size_t order_size);



#ifdef __cplusplus
}
#endif
#endif