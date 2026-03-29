#ifndef TESTER_H
#define TESTER_H

#include "graph.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

int areAlmostEqual(float a, float b);
void testOp(Op op, const int64_t* sh_a, const int64_t* sh_b, const int64_t* sh_c, 
    float* fill_a, float* fill_b, float fill_c, float* unary_out);

#endif