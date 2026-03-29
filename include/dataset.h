#ifndef DATASET_H
#define DATASET_H

#include "utils.h"
#include "probhelper.h"
#include "arena.h"
#include <stdlib.h>
#include <stdalign.h>

typedef enum { DATA_XOR, DATA_TMOONS, DATA_SPIRAL, DATA_FPETALS } DatasetShape;

// class_dpoints allocated linearly by arena as well
// should introduce shapes as well (no problem so far since dims 2 is hardcoded for spiral)
typedef struct {
    float* class_dpoints;
    int num_classes;
    int num_data_points;
    int data_dims;
} Dataset;

void generate_dataset(Dataset* dataset, Arena* arena, int dims, int num_data_points, int num_classes, DatasetShape shape, uint32_t* state);
void free_dataset(Dataset* dataset);
void shuffle_indexes(int* shuffle_arr, int arr_size, uint32_t* rng);

#endif