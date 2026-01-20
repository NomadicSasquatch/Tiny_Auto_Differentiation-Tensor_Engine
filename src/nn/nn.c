#include "nn.h"

static void weight_init_matrix(Tensor* tensor, InitScheme init_scheme) {
    if(!tensor) {
        printf("weight_init_matrix failed, tensor is NULL");
        return;
    }

    int fan_in = (int) tensor->shape[0];
    int fan_out = (int) tensor->shape[1];

    size_t number_elements = total_elems(tensor);

    if(init_scheme == INIT_XAVIER_UNIFORM) {

    }
    else if(init_scheme == INIT_XAVIER_NORMAL) {

    }
    else if(init_scheme == INIT_HE_UNIFORM) {

    }
    else if(init_scheme == INIT_HE_NORMAL) {

    }
    else {
        fatal("weight_init_matrix failed, unknown init_scheme %d", (int) init_scheme);
    }
}

static void init_bias(Tensor* tensor) {
    if(!tensor) {
        printf("init_bias cannot run, bias tensor is NULL");
        return;
    }

    tensor_fill(tensor, 0.0f);   
}

static void linear_init(Linear* layer, Arena* param_arena, int in_features, int out_features, InitScheme init_scheme, uint32_t* rng) {
    if(!layer || !param_arena) {
        prinf("layer_init failed, layer or param_arena is NULL");
        return;
    }

    layer->in_features = in_features;
    layer->out_features = out_features;

    int64_t w_shape = { in_features, out_features };
    int64_t b_shape = { 1, out_features };

    layer->weight = tensor_new(param_arena, 2, w_shape);
    layer->bias = tensor_new(param_arena, 2, b_shape);

    layer->weight->grad = tensor_zeroes_like(param_arena, layer->weight);
    layer->bias->grad = tensor_zeroes_like(param_arena, layer->bias);


}
void init_mlp(MLP* nn, 
            Arena* param_arena,
            int num_layers, 
            int input_dim, 
            int hidden_dim, 
            int output_dim, 
            Activation hidden_activation,
            InitScheme hidden_init, 
            InitScheme output_init, 
            unsigned* rng_state) {

            }
void mlp_forward(Graph* graph, Node* input, const MLP* nn);
void mlp_free(MLP* nn);