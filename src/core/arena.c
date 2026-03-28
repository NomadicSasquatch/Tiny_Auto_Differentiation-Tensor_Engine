#include "arena.h"
#include "utils.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#ifndef MAP_ANONYMOUS
#  ifdef __linux__
#    include <asm-generic/mman-common.h>
#  endif
#endif
#ifndef MAP_ANONYMOUS
#  ifdef MAP_ANON
#    define MAP_ANONYMOUS MAP_ANON
#  else
#    error "MAP_ANONYMOUS is not available on this platform"
#  endif
#endif

void arena_init(Arena* arena, size_t bytes) {
    // pages may be read and written to, as per the PROT flags
    /*
        MAP_PRIVATE means to be creating arena private copy-on-write mapping. Updates to the
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

    if (ptr == MAP_FAILED) {
        fatal("mmap: %s", strerror(errno));
    }

    arena->base = arena->curr = ptr;
    arena->end = arena->base + bytes;
}

void* arena_alloc(Arena* arena, size_t bytes, size_t align) {
    arena->curr = ALIGN_UP(arena->curr, align);

    if (arena->curr + bytes > arena->end) {
        fatal("Out Of Allocated Arena Memory!");
    }

    void* ret = arena->curr;
    arena->curr += bytes;

    return ret;
}

void arena_reset(Arena* arena) {
    arena->curr = arena->base;
}

void arena_free(Arena* arena) {
    if (!arena || !arena->base) return;
    size_t bytes = (size_t)(arena->end - arena->base);
    munmap(arena->base, bytes);
    arena->base = arena->curr = arena->end = NULL;
}

#ifdef ARENA_SELFTEST_MAIN
#include <assert.h>

int main(void) {
    Arena arena;
    arena_init(&arena, 1024);

    void* p1 = arena_alloc(&arena, 16, 8);
    void* p2 = arena_alloc(&arena, 32, 16);

    assert(p1 != NULL);
    assert(p2 != NULL);
    assert(((uintptr_t)p1 % 8) == 0);
    assert(((uintptr_t)p2 % 16) == 0);

    arena_reset(&arena);

    void* p3 = arena_alloc(&arena, 16, 8);
    assert(p3 == p1);

    int *p4 = arena_alloc(&arena, sizeof(int), _Alignof(int));
    *p4 = 123;
    printf("arena selftest ok: %d\n", *p4);

    int *p5 = arena_alloc(&arena, sizeof(int), _Alignof(int));
    *p5 = 456;
    printf("arena selftest ok: %d\n", *p5);

    arena_reset(&arena);
    printf("Has been reset: %d\n", (int)(((void*)arena.curr) == ((void*)p4)));

    arena_free(&arena);
    printf("arena selftest passed\n");
    return 0;
}
#endif