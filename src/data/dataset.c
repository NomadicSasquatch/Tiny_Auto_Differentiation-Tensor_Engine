#include "dataset.h"


void generate_dataset(Dataset* dataset, int num_data_points, int num_classes, DatasetShape shape) {
    dataset->num_classes = num_classes;
    dataset->num_data_points = num_data_points;
    dataset->class_dpoints = (float**) malloc(num_classes * sizeof(float*));

    for(int i = 0; i < num_classes; i++) {
        dataset->class_dpoints[i] = (float*) malloc(num_data_points * sizeof(float));
    }

    if(shape == DATA_XOR) {

    }
    else if(shape == DATA_TMOONS) {

    }
    else if(shape == DATA_SPIRAL) {
        
    }
    else if(shape == DATA_FPETALS) {

    }
    else {
        fatal("generate_dataset cannot run: unkown dataset shape %d", (int) shape);
    }
}

void free_dataset(Dataset* dataset) {
    if(!dataset) {
        return;
    }

    for(int i = 0; i < dataset->num_classes; i++) {
        free(dataset->class_dpoints[i]);
    }

    dataset->class_dpoints = NULL;
    dataset->num_classes = 0;
    dataset->num_data_points = 0;
    free(dataset->class_dpoints);
}