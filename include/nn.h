#ifndef NN_H
#define NN_H

#include "tensor.h"
#include "arena.h"
#include "graph.h"
#include "utils.h"
#include "op.h"

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    INIT_XAVIER_UNIFORM = 0,
    INIT_XAVIER_NORMAL,
    INIT_HE_UNIFORM,
    INIT_HE_NORMAL,
} InitScheme;

typedef enum {
    ACT_NONE = 0,
    ACT_RELU,
    ACT_TANH,
    ACT_SIGMOID
} Activation;

// Abstracting layers for the AD and not the actual nn
typedef struct Linear {
    size_t in_features;
    size_t out_features;
    Tensor* weight;
    Tensor* bias;
} Linear;

typedef struct MLP {
    int num_layers;
    Linear* layers;
    Activation hidden_activation;
} MLP;

// Fully connected always
void init_mlp(MLP* nn, 
            Arena* param_arena,
            int num_layers, 
            int input_dim, 
            int hidden_dim, 
            int output_dim, 
            Activation hidden_activation,
            InitScheme hidden_init, 
            InitScheme output_init, 
            unsigned* rng_state);
void mlp_forward(Graph* graph, Node* input, const MLP* nn);
void mlp_free(MLP* nn);


#ifdef __cplusplus
}
#endif

#endif