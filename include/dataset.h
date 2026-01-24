#ifndef DATASET_H
#define DATASET_H

#include "utils.h"
#include "probhelper.h"
#include <stdlib.h>
#include <stdalign.h>

typedef enum { DATA_XOR, DATA_TMOONS, DATA_SPIRAL, DATA_FPETALS } DatasetShape;

// class_dpoints is going to be num_classes x num_data_points
// Take data shape to be always [num_classes x num_data_points x num_axes]
typedef struct {
    float* class_dpoints;
    int num_data_points;
    int num_classes;
} Dataset;

void generate_dataset(Dataset* dataset, int num_data_points, int num_classes, DatasetShape shape);
void free_dataset(Dataset* dataset);

#endif