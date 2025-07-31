#ifndef ARENA_H
#define ARENA_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/mman.h>

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

// memory arena struct for fast allocation and resets
typedef struct {
    uint8_t* base;
    uint8_t* curr;
    uint8_t* end;
} Arena;


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

    void* ptr = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);

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