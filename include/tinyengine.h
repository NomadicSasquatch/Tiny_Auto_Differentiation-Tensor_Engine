#ifndef TINYENGINE_H
#define TINYENGINE_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
#include <pthread.h>
#include <sys/mman.h>
#include <errno.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>
#include <stdarg.h>

// use C linkage for any of the libraries that are in cpp
#ifdef __cplusplus
extern "C" {
#endif  

// defining logic to break out of program and debug
#define fatal(format, ...)                       \
    do {                                         \
        fprintf(stderr, "Fatal: " format "\n",   \
                ##__VA_ARGS__);                  \
        exit(EXIT_FAILURE);                      \
    } while(0)

// round pointer so that subsequent allocations are aligned(eg cache line) and avoid msialigned access, the alignment a would be taken from alignof()
#define ALIGN_UP(p, a)                           \
        (void*)((((uintptr_t)(p) + ((a)-1)) & ~((uintptr_t)((a)-1))))

// basic operator types/operations for auto differntiator graph later on, mayhapz can add more later
typedef enum { OP_ADD, OP_MUL, OP_MATMUL, OP_RELU, OP_SOFTMAX } Op;

// tensor struct, grad for easier (lazier) backpropagation
// data is all stored within our arena
typedef struct {
    float* data;
    int64_t shape[6];
    int64_t stride[6];
    int ndim;
    struct Tensor* grad;
} Tensor;

// memory arena struct for fast allocation and resets
typedef struct {
    uint8_t* base,
    uint8_t* curr,
    uint8_t* end,
} Arena;

// atomic int to prevent race conditions
// we will be using toposort - topo_index for index in topo order, parents is the atomic counter for Kahn's algo
typedef struct {
    Op operation;
    Tensor* out;
    struct Node* inputs[3];
    int n_input;
    int topo_index;
    atomic_int pending_parents;
    // add children?
} Node;

typedef struct {
    Node** nodes;
    size_t n;
    size_t capacity;
} Graph;

// prototypes for later implementation 
void compute_rowmajor_strides(Tensor *t);
Tensor* tensor_new(Arena *arena, int ndim, const int64_t *shape);

void graph_init(Graph *g);
Tensor* add_node(Graph *g, Op op, int n_in, Node **inputs);
void topo_sort(Graph *g, Node ***out_order, size_t *out_n);
void forward(Graph *g, Node **order, size_t n);
void backward(Graph *g, Node **order, size_t n, Tensor *loss);


static inline void arena_init(Arena* arena, size_t bytes) {
    // pages may be read and written to, as per the PROT flags
    /*
        MAP_PRIVATE means to be creating a private copy-on-write mapping. Updates to the
        mapping are not visible to other processes mapping the same
        file, and are not carried through to the underlying file.
        It is unspecified whether changes made to the file after
        the mmap() call are visible in the mapped region.

        MAP_ANONYMOUS means The mapping is not backed by any file; its contents are
        initialized to zero.  The fd argument is ignored; however,
        some implementations require fd to be -1 if MAP_ANONYMOUS
        (or MAP_ANON) is specified, and portable applications
        should ensure this.  The offset argument should be zero.
        Support for MAP_ANONYMOUS in conjunction with MAP_SHARED
        was added in Linux 2.4.
    */

    void* ptr = mmap(NULL, bytes, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE, -1, 0);

    if(ptr == NULL) {
        fatal("mmap: %s", strerror(errno));
    }

    arena->base = arena->curr = ptr;
    arena->end = arena->base + bytes;
}

static inline void* arena_alloc(Arena* arena, size_t bytes, size_t align) {
    arena->curr = ALIGN_UP(arena->curr, align);

    if(arena->curr + bytes > arena->end) {
        fatal("Out Of Allocated Arena Memory!");
    }

    void* ret = arena->curr;
    arena->curr += bytes;
    
    return ret;
}

static inline void arena_reset(Arena* arena) {
    arena->curr = arena->base;
}





#ifdef __cplusplus
}
#endif
#endif