#include "op.h"
#include "utils.h"

#include <string.h>
#define MAX_OPS 20

static const OpKernel* registry[MAX_OPS];

void register_opkernel(const OpKernel* kernel) {
    if(kernel->optype < 0 || kernel->optype >= MAX_OPS) {
        fatal("Registry is full!");
    }
    registry[kernel->optype] = kernel;
}

const OpKernel* get_opkernel(Op optype) {
    if(optype < 0 || optype >= MAX_OPS) return NULL;
    
    return registry[optype];
}

void testOp(Op op, const int64_t* sh_a, const int64_t* sh_b, const int64_t* sh_c, float fill_a, float fill_b, float fill_c) {
    const OpKernel* k = get_opkernel(op);

    if(!k)  printf("FAIL: op %d not registered\n", (int)op);
    else printf("PASS: op %d is registered\n", (int)op);

    if(!k->forward) printf("FAIL: op %s has NULL forward\n", k->name);
    else printf("PASS: op %s has valid forward\n", k->name);

    if(!k->backward) printf("FAIL: op %s has NULL backward\n", k->name);
    else printf("Pass: op %s has valid backward\n", k->name);

    Arena arena;
    arena_init(&arena, 1024);
    Graph graph;
    graph_init(&graph, &arena);

    Tensor* a = tensor_new(&arena, 2, sh_a);
    tensor_fill(a, fill_a);
    Tensor* b = tensor_new(&arena, 2, sh_b);
    tensor_fill(b, fill_b);
    Tensor* out = tensor_new(&arena, 2, sh_c);
    tensor_fill(out, 0);

    Node* input1 = graph_add_input(&graph, a);
    Node* input2 = graph_add_input(&graph, b);
    Node** inputs = malloc(2 * sizeof(Node*));
    inputs[0] = input1;
    inputs[1] = input2;

    Node* output = add_node(&graph, op, 2, inputs);
    Tensor* check = output->out;
    size_t total = total_elems(check);

    k->forward(output);
    printf("Input Tensor 1:\n");
    print_tensor(a);
    printf("\nInput Tensor 2:\n");
    print_tensor(b);
    printf("\nOuptut tensor:\n");
    print_tensor(check);

    for(size_t i = 0; i < total; i++) {
        assert(check->data[i] == fill_c);
    }
    printf("%s selftest passed\n", k->name);
    arena_free(&arena);
    free(inputs);
}