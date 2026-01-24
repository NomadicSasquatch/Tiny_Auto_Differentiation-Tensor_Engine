#include "loss.h"

float cross_entropy(const float* probs, int y_class) {
    // Does not have checks for y_class being out of idnex but it should not be a problem :)
    float y_hat = probs[y_class];

    if(y_hat < 1e-12f) {
        y_hat = 1e-12f;
    }

    return -logf(y_hat);
}