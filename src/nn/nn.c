#include "nn.h"
#include "optim.h"

// rng for random init
static void weight_init_matrix(Tensor* tensor, InitScheme init_scheme, uint32_t* rng) {
    if(!tensor) {
        printf("weight_init_matrix failed, tensor is NULL");
        return;
    }

    int fan_in = (int) tensor->shape[0];
    int fan_out = (int) tensor->shape[1];

    size_t number_elements = total_elems(tensor);

    if(init_scheme == INIT_XAVIER_UNIFORM) {
        float a = sqrtf(6.0f / (float) (fan_in + fan_out));

        for(size_t i = 0; i < number_elements; i++) {
            tensor->data[i] = rand_uniform(rng, -a, a);
        }

        return;
    }
    else if(init_scheme == INIT_XAVIER_NORMAL) {
        float std = sqrt(2.0f / (float) (fan_in + fan_out));

        for(size_t i = 0; i < number_elements; i++) {
            tensor->data[i] = rand_normal(rng, 0.0f, std);
        }

        return;
    }
    else if(init_scheme == INIT_HE_UNIFORM) {
        float a = sqrtf(6.0f / (float)(fan_in));

        for(size_t i = 0; i < number_elements; i++) {
            tensor->data[i] = rand_uniform(rng, -a, a);
        }

        return;
    }
    else if(init_scheme == INIT_HE_NORMAL) {
        float std = sqrtf(2.0f / (float)(fan_in));

        for(size_t i = 0; i < number_elements; i++) {
            tensor->data[i] = rand_normal(rng, 0.0f, std);
        }

        return;
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

static void linear_init(Linear* layer, Arena* param_arena, size_t in_features, size_t out_features, InitScheme init_scheme, uint32_t* rng) {
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

    weight_init_matrix(layer->weight, init_scheme, rng);
    init_bias(layer->bias);

    // Sanity check to zero gradients and bias? Doubt its needed for now, zeroed in tensor_zeroes_like
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
            uint32_t* rng_state) {
                if(!nn || ! param_arena) {
                    fatal("init_mlp cannot run: nn or param_arena is NULL");
                }
                if(num_layers < 1) {
                    fatal("init_mlp cannot run: num_layers must be >= 1, curr num_layers = %d", num_layers);
                }
                if(hidden_dim < 1) {
                    fatal("init_mlp cannot run: hidden_dim must be >= 1, curr hidden_dim = %d", hidden_dim);
                }

                nn->num_layers = num_layers;
                nn->hidden_activation = hidden_activation;
                // Arena?
                nn->layers = (Linear*) malloc((size_t) num_layers * sizeof(Linear));

                if(num_layers == 1) {
                    linear_init(nn->layers, param_arena, input_dim, output_dim, output_init, rng_state);

                    return;
                }

                linear_init(&nn->layers[0], param_arena, input_dim, hidden_dim, hidden_init, rng_state);

                for(int i = 1; i < num_layers - 1; i++) {
                    linear_init(&nn->layers[i], param_arena, hidden_dim, hidden_dim, hidden_init, rng_state);
                }

                linear_init(&nn->layers[num_layers-1], param_arena, hidden_dim, output_dim, output_init, rng_state);

                return;
            }

Node* layer_forward(Graph* graph, Node* input, const Linear* layer) {
    if(!graph || !input || !layer) {
        fatal("mlp_forward cannot run: graph or input or nn is NULL");
    }

    Tensor* weight = layer->weight;
    Tensor* bias = layer->bias;

    Node* w_node = graph_add_input(graph, weight);
    Node* inputs_w[2] = { input, w_node };
    Node* mm_node = add_node(graph, OP_MATMUL, 2, inputs_w);

    Node* b_node = graph_add_input(graph, bias);
    Node* inputs_b[2] = { mm_node, b_node };
    Node* a_node = add_node(graph, OP_ADD, 2, inputs_b);

    return a_node;
}

Node* apply_activation(Graph* graph, Activation activation, Node* input) {
    if(activation == ACT_NONE) {
        return input;
    }

    Node* act_input[1] = { input };
    Op op_type;

    if(activation == ACT_RELU) {
        op_type = OP_RELU;
    }
    else if(activation == ACT_SIGMOID) {
        op_type = OP_SIGMOID;
    }
    else if(activation == ACT_TANH) {
        op_type = OP_TANH;
    }
    else if(activation == ACT_SOFTMAX) {
        op_type = OP_SOFTMAX;
    }
    else {
        fatal("forward activation cannot run: activation not recognised, activation index: %d", (int) activation);
    }

    Node* output = add_node(graph, op_type, 1, act_input);

    return output;
}

Node* mlp_forward(Graph* graph, Node* input, const MLP* nn) {
    if(!graph || !input || !nn) {
        fatal("mlp_forward cannot run: graph or input or nn is NULL");
    }

    Node* head = input;

    for(int i = 0; i < nn->num_layers - 1; i++) {
        head = layer_forward(graph, head, &nn->layers[i]);
        head = apply_activation(graph, nn->hidden_activation, head);
    }

    head = layer_forward(graph, head, &nn->layers[nn->num_layers-1]);

    // Raw logits, final output activation yet to be applied
    return head;
}

static void mlp_zero_grads(MLP* nn) {
    for(int l = 0; l < nn->num_layers; l++) {
        tensor_zero_grad(nn->layers[l].weight);
        tensor_zero_grad(nn->layers[l].bias);
    }
}

void mlp_free(MLP* nn) {
    if(!nn) {
        return;
    }

    free(nn->layers);
    nn->layers = NULL;
    nn->layers = 0;
}