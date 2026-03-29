#include "op.h"
#include "utils.h"

#include <string.h>
#include <math.h>

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

static int areAlmostEqual(float a, float b) {
    float epsilon = 0.00001f;
    return fabs(a - b) < epsilon;
}

void testOp(Op op, const int64_t* sh_a, const int64_t* sh_b, const int64_t* sh_c, 
    float* fill_a, float* fill_b, float fill_c, float* unary_out) {
    const OpKernel* k = get_opkernel(op);

    if(!k)  printf("FAIL: op %d not registered\n", (int)op);
    else printf("PASS: op %d is registered\n", (int)op);

    if(!k->forward) printf("FAIL: op %s has NULL forward\n", k->name);
    else printf("PASS: op %s has valid forward\n", k->name);

    if(!k->backward) printf("FAIL: op %s has NULL backward\n", k->name);
    else printf("Pass: op %s has valid backward\n", k->name);

    Arena arena;
    arena_init(&arena, 4096);
    Graph graph;
    graph_init(&graph, &arena);

    Tensor* a, *b, *out;
    Node** inputs;
    int n_in;

    if(!unary_out) {
        a = tensor_new(&arena, 2, sh_a);
        tensor_fill(a, fill_a[0]);
        b = tensor_new(&arena, 2, sh_b);
        tensor_fill(b, fill_b[0]);
        out = tensor_new(&arena, 2, sh_c);
        tensor_fill(out, 0);

        inputs = malloc(2 * sizeof(Node*));
        Node* input1 = graph_add_input(&graph, a);
        Node* input2 = graph_add_input(&graph, b);
        inputs[0] = input1;
        inputs[1] = input2;
        n_in = 2;
    }
    else {
        a = tensor_new(&arena, 2, sh_a);
        size_t total = total_elems(a);
        size_t i = 0;

        for(; i < total/2; i++) {
            a->data[i] = fill_a[0];
        }
        for(; i < total; i++) {
            a->data[i] = fill_b[0];
        }

        inputs = malloc(1 * sizeof(Node*));
        Node* input1 = graph_add_input(&graph, a);
        inputs[0] = input1;
        n_in = 1;
    }

    Node* output = add_node(&graph, op, n_in, inputs);
    Tensor* check = output->out;
    size_t total = total_elems(check);

    k->forward(output);
    printf("\nForward Test:\n");
    printf("\nInput Tensor 1:\n");
    print_tensor(a);
    if(!unary_out) {
        printf("\nInput Tensor 2:\n");
        print_tensor(b);
    }
    printf("\nOuptut tensor:\n");
    print_tensor(check);

    for(size_t i = 0; i < total; i++) {
        if(unary_out) assert(areAlmostEqual(check->data[i], unary_out[i]));
        else assert(areAlmostEqual(check->data[i], fill_c));
    }

    graph_ensure_grad(&graph, output->inputs[0]->out);
    tensor_fill(output->inputs[0]->out->grad, fill_a[0]);
    graph_ensure_grad(&graph, output->out);

    if(!unary_out) {
        graph_ensure_grad(&graph, output->inputs[1]->out);
        tensor_fill(output->inputs[1]->out->grad, fill_b[0]);
    }
    tensor_fill(output->out->grad, fill_c);

    printf("%s forward selftest passed\n", k->name);

    printf("\nBackward Test:\n");
    printf("\nInput Tensor 1 Grad:\n");
    print_tensor(a->grad);
    if(!unary_out) {
        printf("\nInput Tensor 2 Grad:\n");
        print_tensor(b->grad);
    }

    k->backward(output);

    printf("\nInput Tensor 1 Grad:\n");
    print_tensor(a->grad);
    if(!unary_out) {
        printf("\nInput Tensor 2 Grad:\n");
        print_tensor(b->grad);
    }


    for(size_t i = 0; i < total; i++) {
        if(unary_out) {
            if(i < total/2) assert(areAlmostEqual(a->grad->data[i], fill_a[1]));
            else assert(areAlmostEqual(a->grad->data[i], fill_b[1]));
        }
        else {
            assert(areAlmostEqual(output->inputs[0]->out->grad->data[i], fill_a[1]));
            assert(areAlmostEqual(output->inputs[1]->out->grad->data[i], fill_b[1]));
        }
    }

    printf("%s backward selftest passed\n", k->name);

    arena_free(&arena);
    free(inputs);
}