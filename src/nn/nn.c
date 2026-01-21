#include "nn.h"
#include "optim.h"

/*XORShift is just a fast (but weak) PRNG that is a subset of LFSRs. Deterministic and not truly random,. 
  Returns a 32 bit int */
static uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;

    // Avoiding 0 edgecase (all 0s still after shift)
    if(x == 0) {
        x = 0x12345678u;
    }

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;

    return x;
}

static float rand_uniform01(uint32_t* state) {
    uint32_t tmp = xorshift32(state);

    // Normalise to [0,1), instead of [0,1] by dividing by (float) UINT32_MAX to avoid edgecases
    return (float)tmp * (1.0f / 4294967296.0f);
}

// Returns random float in [low,high], for kaiming
static float rand_uniform(uint32_t* state, float low, float high) {
    return low + (high - low) * rand_uniform01(state);
}

/* Generates a float sampled from a Gaussian distribution of X ~ N(mean, std^2), using
   box muller transform*/
/* Summary for box muller transform:
    -> Transforms two independent numbers sampled from uniform distribution U1, U2 ~ Uniform(0,1) to
        two independent numbers from a standard normal random numbres Z1, Z2 ~ N(0,1)
    -> Picking a random point (x, y) from a 2D Gaussian centered at the origin, the angle around the origin
        is uniform(no direction is preferred) and the distance from the origin is not uniform (most points
        cluster around the center)
    -> Therfore we can generate a Gaussian point by generating a random angle _theta_, random radius _r_ and
        convert the from polar to cartesian where x = _r_cos(_theta_) and y = _r_sin(_theta_)
    -> Box mullers job is to pick the correct radius distribution and pick a uniform angle
    -> _theta_ must be uniform on (0, 2*pi), so if U2 ~ Uniform(0,1), then _theta_ = 2 * pi * U2
    -> Raidus was obtained by solving from the conversion of uniform distribution to Rayleigh using inverse
        CDF 
    -> Convert to cartesian from polar
    -> Distribution is normal, since the geometry of a 2D Gaussian is recreated, with the angle uniform
        from 2 * pi * U2 and radius having rayleigh distribution from sqrt(-2ln(U1))
    -> We can then convert this standard normal to any normal with mean mu and std sigma */
    
static float rand_normal(uint32_t* state, float mean, float std) {
    float u1 = rand_uniform01(state);
    float u2 = rand_uniform01(state);

    // To avoid log(0)
    if(u1 < 1e-12f){
        u1 = 1e-12f;
    }

    float r = sqrtf(-2.0f * logf(u1));
    // 2 * pi
    float theta = 2.0f * 3.14159265358979323846f * u2;
    float z = r * cosf(theta);

    return mean + std * z;
}

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
            unsigned* rng_state) {
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

void mlp_free(MLP* nn) {
    if(!nn) {
        return;
    }

    free(nn->layers);
    nn->layers = NULL;
    nn->layers = 0;
}