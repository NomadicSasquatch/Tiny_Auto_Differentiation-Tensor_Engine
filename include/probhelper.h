#ifndef PROBHELPER_H
#define PROBHELPER_H

#include <stdint.h>
#include <math.h>
#ifndef PI
#define PI 3.14159265358979323846
#endif

static uint32_t xorshift32(uint32_t* state);
static float rand_uniform01(uint32_t* state);
static float rand_uniform(uint32_t* state, float low, float high);
static float rand_normal(uint32_t* state, float mean, float std);

#endif