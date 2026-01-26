#ifndef PROBHELPER_H
#define PROBHELPER_H

#include <stdint.h>
#include <math.h>
#ifndef PI
#define PI 3.14159265358979323846
#endif

uint32_t xorshift32(uint32_t* state);
float rand_uniform01(uint32_t* state);
float rand_uniform(uint32_t* state, float low, float high);
float rand_normal(uint32_t* state, float mean, float std);

#endif