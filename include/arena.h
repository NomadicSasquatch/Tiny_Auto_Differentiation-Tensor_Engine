#ifndef ARENA_H
#define ARENA_H

#include <stddef.h>
#include <stdint.h>

// use C linkage for any of the libraries that are in cpp
#ifdef __cplusplus
extern "C" {
#endif

/* One persistent arena, one scratch arena, one data arena. Persistent arena is for values that do not change per iteration/epoch like weights, biases, weight grad and bias grad.
Scratch arena stores intermediate activation tensors, current input and output tensors x and y, output loss tensor, intermediate tensors produced by ops. */

// memory arena struct for fast allocation and resets
typedef struct Arena {
    uint8_t* base;
    uint8_t* curr;
    uint8_t* end;
} Arena;

void arena_init(Arena* arena, size_t bytes);
void* arena_alloc(Arena* arena, size_t bytes, size_t align);
void arena_reset(Arena* arena);
void arena_free(Arena* arena);

#ifdef __cplusplus
}
#endif

#endif