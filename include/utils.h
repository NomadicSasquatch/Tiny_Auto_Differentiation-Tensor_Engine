#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// defining logic to break out of program and debug
// Using a function avoids -Wpedantic warnings for empty __VA_ARGS__ cases.
void fatalf(const char* line, int line, const char* format, ...);
#define fatal(...) fatalf(__FILE__, __LINE__, __VA_ARGS__)

// round pointer so that subsequent allocations are aligned(eg cache line) and avoid msialigned access, the alignment a would be taken from alignof()
#define ALIGN_UP(p, a)                           \
        (void*)((((uintptr_t)(p) + ((a)-1)) & ~((uintptr_t)((a)-1))))

#endif