#ifdef ARENA_SELFTEST_MAIN
#include <stdio.h>
#include "arena.h"

int main(void) {
    Arena a;
    arena_init(&a, 4096);

    int *p = arena_alloc(&a, sizeof(int), _Alignof(int));
    *p = 123;
    printf("arena selftest ok: %d\n", *p);

    int *p2 = arena_alloc(&a, sizeof(int), _Alignof(int));
    *p2 = 456;
    printf("arena selftest ok: %d\n", *p2);

    arena_reset(&a);
    printf("Has been reset: %d\n", (int)(((void*)a.curr) == ((void*)p)));

    arena_free(&a);
    return 0;
}

#endif