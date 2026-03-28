#include "utils.h"
#include <stdarg.h>

void fatalf(const char* file, int line, const char* fmt, ...) {
    fprintf(stderr, "Fatal (%s:%d): ", file, line);

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fputc('\n', stderr);
    exit(EXIT_FAILURE);
}

#ifdef UTILS_SELFTEST_MAIN
#include <assert.h>

int main(void) {
    char buf[64];
    void *p = buf + 3;
    void *a = ALIGN_UP(p, 8);
    uintptr_t pa = (uintptr_t)a;
    uintptr_t pp = (uintptr_t)p;

    assert(pa % 8 == 0);
    assert(pa >= pp);
    assert(pa - pp < 8);

    void *a2 = ALIGN_UP(a, 8);
    assert((uintptr_t)a2 == (uintptr_t)a);

    printf("ALIGN_UP tests passed\n");
    fatal("fatal error");
    return 0;
}

#endif