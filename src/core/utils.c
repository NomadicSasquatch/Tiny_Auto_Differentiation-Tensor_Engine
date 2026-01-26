#include "tinyengine.h"
#include "utils.h"

void fatalf(const char* file, int line, const char* fmt, ...) {
    fprintf(stderr, "Fatal (%s:%d): ", file, line);

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fputc('\n', stderr);
    exit(EXIT_FAILURE);
}