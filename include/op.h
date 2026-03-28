#include "graph.h"
#include "tensor.h"
#include "utils.h"

#include <assert.h>

#ifndef OP_H
#define OP_H

typedef struct Node Node;
// Typedef but with a function pointer, for us to use in our OpKernel, to call the actual operations easier
// Format: typedef {return_type} (*{function_name})({parameter_list});
typedef void (*OpForward)(Node*);
typedef void (*OpBackward)(Node*);
typedef enum { OP_INPUT, OP_ADD, OP_SUB, OP_MUL, OP_MATMUL, OP_RELU, OP_SOFTMAX, OP_SIGMOID, OP_TANH } Op;

typedef struct {
    Op optype;
    const char* name;
    OpForward forward;
    OpBackward backward;
} OpKernel;

// Called by the op init fns
void register_opkernel(const OpKernel* kernel);

// Get opkernel by its enum
const OpKernel* get_opkernel(Op optype);
// Simple testing function for each op, all 3 nodes are n_dim = 2
// fill_(a,b,c) is the shape of input a,b and output c respectively
// fill_a is the float to fill the first input's entire 2x3 tensor
// fill_b is the float to fill the second output's entire 2x3 tensor
// fill_c is the float that's expected to populate the entire output node's tensor
// the order of operation is assert a OP b = c
// unary_out is the float array to compare with a unary output tensor's data(from 0 to total_elems), if its NULL its not a unary op
// in a unary op, fill_a is the input's first row values, fill_b is the input's second row values
void testOp(Op op, const int64_t* sh_a, const int64_t* sh_b, const int64_t* sh_c, 
    float fill_a, float fill_b, float fill_c, float* unary_out);
#endif