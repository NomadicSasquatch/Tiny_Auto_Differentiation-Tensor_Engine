#ifndef OP_H
#define OP_H

#include <stdlib.h>
#include <stdio.h>

struct Node;
// Typedef but with a function pointer, for us to use in our OpKernel, to call the actual operations easier
// Format: typedef {return_type} (*{function_name})({parameter_list});
typedef void (*OpForward)(Node*);
typedef void (*OpBackward)(Node*);
typedef enum { OP_ADD, OP_MUL, OP_MATMUL, OP_RELU, OP_SOFTMAX } Op;

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

#endif