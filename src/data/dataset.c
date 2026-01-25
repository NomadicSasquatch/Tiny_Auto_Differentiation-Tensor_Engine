#include "dataset.h"
#include "probhelper.h"
#include "arena.h"


void generate_dataset(Dataset* dataset, Arena* arena, int data_dims, int num_data_points, int num_classes, DatasetShape shape, uint32_t* state) {
    dataset->num_classes = num_classes;
    dataset->num_data_points = num_data_points;
    dataset->class_dpoints = arena_alloc(arena, data_dims * sizeof(float) * num_classes * num_data_points, alignof(float));

    if(shape == DATA_XOR) {

    }
    else if(shape == DATA_TMOONS) {

    }
    else if(shape == DATA_SPIRAL) {
        // We only take 2D for now
        if(data_dims != 2) {
            print("DATA_SPIRAL only allows for 2D plane!");
            
            return;
        }
        // t_max (double) max parameter value (controls spiral length ie how many turns), take it as [2 pi, 4 pi] 
        // b (double) radial scale (r = b*t), take it as [0.5, 1.5], with bigger b == more spacing between spirals == easier
        // offset helps with even turning angles per class
        
        // Not dynamic for now, for future changes
        double t_max = 2 * PI;
        double b = 1.0;
        for(int i = 0; i < dataset->num_classes; i++) {
            double offset = 2.0 * PI * (double) i / (double) num_classes;

            for(int j = 0; j < dataset->num_data_points; j++) {
                double t = t_max * rand_uniform01(state);
                double r = b * t;
                double theta = t + offset;

                double x = r * cos(theta);
                double y = r * sin(theta);

                // Std is hardcoded for now, for future chagnes
                x += rand_normal(state, 0, 0.2);
                y += rand_normal(state, 0, 0.2);

                int tmp_idx = data_dims * (i * num_data_points + j);
                dataset->class_dpoints[tmp_idx] = x;
                dataset->class_dpoints[tmp_idx + 1] = y;
            }
        }
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

    dataset->class_dpoints = NULL;
    dataset->num_classes = 0;
    dataset->num_data_points = 0;
    // free arena
}

// Fisher-Yates shuffle, again like in pathfinder repo :^)
void shuffle_indexes(int* shuffle_arr, int arr_size, uint32_t* rng) {
    for(int i = arr_size - 1; i > 0; i--) {
        int j = (int)(rand_uniform01(rng) * (float)(i + 1));
        int tmp = shuffle_arr[i];
        shuffle_arr[i] = shuffle_arr[j];
        shuffle_arr[j] = tmp;
    }
}