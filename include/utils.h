#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

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

#endif