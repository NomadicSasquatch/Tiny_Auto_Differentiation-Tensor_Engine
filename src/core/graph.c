#include "tinyengine.h"
#include "op.h"

// graph capacity is hardcoded to a small value initially first, reinitialisation is done through realloc if size >= capacity
void graph_init(Graph* graph) {
    graph->size = 0;
    graph->capacity = 16;
    graph->nodes = malloc(graph->capacity * sizeof(Node*));
}

Tensor* add_node(Graph* graph, Op op, int n_inputs, Node** inputs) {
    if(graph->size == graph->capacity) {
        graph->capacity *= 2;
        graph->nodes = realloc(graph->nodes, graph->capacity * sizeof(Node*));
    }

    Node* node = malloc(sizeof(Node));
    node->operation = op;
    node->n_input = n_inputs;
    
    for(int i = 0; i < n_inputs; i++) {
        node->inputs[i] = inputs[i];
    }
    // TODO: determine the shape
    int64_t out_shape = {0};
    // TODO: where should arena be, fix dimensions
    node->out = tensor_new(/*arena here*/, inputs[0]->out->data, /*n_dimensions*/, out_shape);
    atomic_init(&node->pending_parents, 0);
    graph->nodes[graph->size++] = node;

    return node->out;
}