#include "probhelper.h"

/*XORShift is just a fast (but weak) PRNG that is a subset of LFSRs. Deterministic and not truly random,. 
  Returns a 32 bit int */
uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;

    // Avoiding 0 edgecase (all 0s still after shift)
    if(x == 0) {
        x = 0x12345678u;
    }

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;

    return x;
}

float rand_uniform01(uint32_t* state) {
    uint32_t tmp = xorshift32(state);

    // Normalise to [0,1), instead of [0,1] by dividing by (float) UINT32_MAX to avoid edgecases
    return (float)tmp * (1.0f / 4294967296.0f);
}

// Returns random float in [low,high], for kaiming
float rand_uniform(uint32_t* state, float low, float high) {
    return low + (high - low) * rand_uniform01(state);
}

/* Generates a float sampled from a Gaussian distribution of X ~ N(mean, std^2), using
   box muller transform*/
/* Summary for box muller transform:
    -> Transforms two independent numbers sampled from uniform distribution U1, U2 ~ Uniform(0,1) to
        two independent numbers from a standard normal random numbres Z1, Z2 ~ N(0,1)
    -> Picking a random point (x, y) from a 2D Gaussian centered at the origin, the angle around the origin
        is uniform(no direction is preferred) and the distance from the origin is not uniform (most points
        cluster around the center)
    -> Therfore we can generate a Gaussian point by generating a random angle _theta_, random radius _r_ and
        convert the from polar to cartesian where x = _r_cos(_theta_) and y = _r_sin(_theta_)
    -> Box mullers job is to pick the correct radius distribution and pick a uniform angle
    -> _theta_ must be uniform on (0, 2*pi), so if U2 ~ Uniform(0,1), then _theta_ = 2 * pi * U2
    -> Raidus was obtained by solving from the conversion of uniform distribution to Rayleigh using inverse
        CDF 
    -> Convert to cartesian from polar
    -> Distribution is normal, since the geometry of a 2D Gaussian is recreated, with the angle uniform
        from 2 * pi * U2 and radius having rayleigh distribution from sqrt(-2ln(U1))
    -> We can then convert this standard normal to any normal with mean mu and std sigma */
    
float rand_normal(uint32_t* state, float mean, float std) {
    float u1 = rand_uniform01(state);
    float u2 = rand_uniform01(state);

    // To avoid log(0)
    if(u1 < 1e-12f){
        u1 = 1e-12f;
    }

    float r = sqrtf(-2.0f * logf(u1));
    // 2 * pi
    float theta = 2.0f * 3.14159265358979323846f * u2;
    float z = r * cosf(theta);

    return mean + std * z;
}