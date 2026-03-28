#include "op.h"
#include "graph.h"
#include "tensor.h"
#include "utils.h"

#include <stddef.h>

static void add_fwd(Node* node) {
    Tensor* A = node->inputs[0]->out;
    Tensor* B = node->inputs[1]->out;
    Tensor* C = node->out;
    size_t number_elements = total_elems(C);

    for(size_t i = 0; i < number_elements; i++) {
        C->data[i] = A->data[i] + B->data[i];
    }
}

static void add_bwd(Node* node) {
    Tensor* gA = node->inputs[0]->out->grad;
    Tensor* gB = node->inputs[1]->out->grad;
    Tensor* gC = node->out->grad;
    size_t number_elements = total_elems(gC);
    
    for(size_t i = 0; i < number_elements; i++) {
        gA->data[i] += gC->data[i];
        gB->data[i] += gC->data[i];
    }
}

static const OpKernel add_kernel = {
    .optype = OP_ADD,
    .name = "add",
    .forward = add_fwd,
    .backward = add_bwd,
};

// Run before main is called
__attribute__((constructor))
static void register_add_kernel(void) {
    register_opkernel(&add_kernel);
}

#ifdef ADD_SELFTEST_MAIN
#include <assert.h>
#include <stdio.h>

int main(void) {
    const OpKernel* k = get_opkernel(OP_ADD);

    if(!k)  printf("FAIL: op %d not registered\n", (int)OP_ADD);
    else printf("PASS: op %d is registered\n", (int)OP_ADD);

    if(!k->forward) printf("FAIL: op %s has NULL forward\n", k->name);
    else printf("PASS: op %s has valid forward\n", k->name);

    if(!k->backward) printf("FAIL: op %s has NULL backward\n", k->name);
    else printf("Pass: op %s has valid backward\n", k->name);

    Arena arena;
    arena_init(&arena, 1024);
    Graph graph;
    graph_init(&graph, &arena);

    const int64_t t_shape[2] = {2, 3};

    Tensor* a = tensor_new(&arena, 2, t_shape);
    tensor_fill(a, 2.0);
    Tensor* b = tensor_new(&arena, 2, t_shape);
    tensor_fill(b, 3.0);
    Tensor* out = tensor_new(&arena, 2, t_shape);
    tensor_fill(out, 0);

    Node* input1 = graph_add_input(&graph, a);
    Node* input2 = graph_add_input(&graph, b);
    Node** inputs = malloc(2 * sizeof(Node*));
    inputs[0] = input1;
    inputs[1] = input2;

    Node* output = add_node(&graph, OP_ADD, 2, inputs);
    Tensor* check = output->out;
    size_t total = total_elems(check);

    k->forward(output);
    print_tensor(check);

    for(size_t i = 0; i < total; i++) {
        assert(check->data[i] == 5);
    }
    printf("add selftest passed\n");
    arena_free(&arena);
    free(inputs);

    return 0;
}
#endif