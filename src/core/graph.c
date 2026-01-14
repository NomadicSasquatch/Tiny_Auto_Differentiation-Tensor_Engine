#include "tinyengine.h"
#include "graph.h"
#include "op.h"

// graph capacity is hardcoded to a small value initially first, reinitialisation is done through realloc if size >= capacity
void graph_init(Graph* graph) {
    graph->size = 0;
    graph->capacity = 16;
    graph->nodes = malloc(graph->capacity * sizeof(Node*));
}

// this part is a little sus, since there is no check with the arena at all
static void graph_size_validity_check(Graph* graph) {
    if(graph->size < graph->capacity) {
        return;
    }

    graph->capacity = (graph->capacity == 0)? 16 : 2 * graph->capacity;
    graph->nodes = (Node**)realloc(g->nodes, g->capacity * sizeof(Node*));

    if(!graph->nodes) {
        fatal("graph_size_validity_check cannot run: realloc failed");
    }
}

void graph_free(Graph* graph) {
    if(!graph) {
        return;
    }

    for(size_t i = 0; i < graph->size; i++) {
        Node* tmp_node = graph->nodes[i];

        if(!tmp->node) {
            continue;
        }
        free(tmp_node->inputs);
        free(tmp_node);
    }

    free(graph->nodes);
    memset(graph, 0, sizeof(*graph));
}

Node* graph_add_input(Graph* graph, Tensor* tensor) {
    if(!graph || !tensor) {
        return NULL;
    }

    graph_size_validity_check(graph);
    Node* node = (Node*)calloc(1, sizeof(Node));

    if(!node) {
        fatal("graph_add_input cannot run: calloc failed");
    }

    node->operation = OP_INPUT;
    node->out = tensor;
    node->inputs = NULL;
    node->n_input = 1;
    node->topo_index = 0;
    atomic_init(&node->pending_parents, 0);

    graph->nodes[g->size++] = node;

    return node;
}

static void ensure_same_shape(const Tensor* a, const Tensor* b) {
    if(a->ndim != b->ndim) {
        fatal("shape mismatch: %d ndim of left arg tensor vs %d ndim of right arg tensor", a->ndim, b->ndim);
    }

    for(int i = 0; i < a->ndim; i++) {
        if(a->shape[i] != b->shape[i]) {
            fatal("shape mismatch at dim %d: %d ndim of left arg tensor vs %d ndim of right arg tensor", i, a->ndim, b->ndim)
        }
    }
}

static Tensor* infer_and_alloc_output(Graph* graph, Op op, int n_inputs, Node** inputs) {
    if(!g) {
        fatal("infer_and_alloc_output cannot run: graph is NULL");
    }
    if(!graph->arena) {
        fatal("infer_and_alloc_output cannot run: arena is NULL");
    }

    Tensor* A = inputs[0]->out;
    Tensor* B = inputs[1]->out;

    if(op == OP_ADD || op == OP_MUL) {
        ensure_same_shape(a, b);

        return tensor_new(graph->arena, A->ndim, A->shape)
    }
    else if(op == OP_MATMUL) {
        if(A->ndim != 2 || B->ndim !=2) {
            fatal("infer_and_alloc_output cannot run: matmul must involve 2 dimensional tensors");
        }

        int64_t ad1 = A->shape[0], ad2 = A->shape[1];
        int64_t bd1 = B->shape[0], bd2 = B->shape[1];

        if(ad2 != bd1) {
            fatal("infer_and_alloc_output cannot run: matmul shape mismatch, %d vs %d", ad2, bd1);
        }

        int64_t output_shape[2] = {ad1, bd2};
        
        return new_tensor(graph->arena, 2, output_shape);
    }

    fatal("infer_and_alloc_output cannot run: OP type index (%d) is not supported", (int) op);

    return NULL;
}

Tensor* add_node(Graph* graph, Op op, int n_inputs, Node** inputs) {
    if(!graph || !inputs) return NULL;
    if(n_inputs <= 0) fatal("add_node: n_inputs must be > 0");
    if(op == OP_INPUT) fatal("add_node: OP_INPUT is reserved for leaves");

    graph_grow_if_needed(graph);

    Node* output_node = malloc(sizeof(Node));

    if(!output_node) {
        fatal("add_node cannot run: malloc failed for new node");
    }

    output_node->operation = op;
    output_node->n_input = n_inputs;
    output_node->inputs = (Node*) malloc((size_t) n_inputs * sizeof(Node*));

    if(!output_node->inputs) {
        fatal("add_node cannot run: malloc for output_node->inputs failed");
    }

    for(int i = 0; i < n_inputs; i++) {
        output_node->inputs[i] = inputs[i];
    }

    output_node->out = infer_and_alloc_output(graph, op, n_inputs, inputs);
    atomic_init(&output_node->pending_parents, 0);
    graph->nodes[graph->size++] = output_node;

    return output_node->out;
}

void topological_sort(Graph* graph, Node** output_order, size_t* total_outputs) {
    if(!graph || !output_order || !total_outputs) {
        printf("topological sort error: one of the inputs is NULL");

        return NULL;
    }

    size_t total_nodes = graph->size;
    size_t order_idx = 0;
    Node** order = (Node*) malloc(total_nodes * sizeof(Node*));
    
    if(!queue) {
        fatal("topological sort cannot run: order malloc failed");
    }

    int* in_degree = calloc(total_nodes, sizeof(int));

    if(!in_degree) {
        fatal("topological sort cannot run: in_degree calloc failed");
    }

    size_t head = 0, tail = 0;
    size_t* queue = (size_t) malloc(total_nodes * sizeof(size_t));
    
    if(!queue) {
        fatal("topological sort cannot run: queue malloc failed");
    }

    for(size_t i = 0; i < total_nodes; i++) {
        graph->nodes[i]->topo_index = (int) i;

        Node* tmp_node = graph->nodes[i];
        in_degree[i] = tmp_node->n_inputs;
    }

    for(size_t i = 0; i < total_nodes; i++) {
        if(in_degree[i] == 0) {
            queue[tail++] = i;
        }
    }

    while(head < tail) {
        size_t src = queue[head++];
        Node* src_node = graph->nodes[src];
        order[order_idx++] = src_node;

        for(size_t i = 0; i < total_nodes; i++) {
            Node* dst_node = graph->nodes[i];
            for(int j = 0; j < dst_node->n_inputs; j++) {
                if(dst_node->inputs[j] == src_node) {
                    if(--in_degree[i] == 0) {
                        queue[tail++] = i;
                    }   
                    // break;
                }
            }
        }
    }

    free(queue);
    free(in_degree);

    if(order_idx != total_nodes) {
        free(order);
        fatal("topological sort cannot run: graph has a cycle or disconnected nodes");
    }

    *output_order = order;
    *total_outputs = total_nodes;
}

void forward() {

}

void calc_grad() {

}