#ifndef MODEL_H
#define MODEL_H

#include "nn.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void run_train(const MLP* nn);
void run_inference(const MLP* nn);

static void save_model(const char* file_path, MLP* nn) {
    FILE* f = fopen(file_path, "wb");
    if(!f) {
        fatal("save_model: failed to open %s", file_path);
    }

    const char header[8] = "TMLP000";
    fwrite(header, 1, 8, f);

    // Perhaps should include other metadata of the MLP, (to add in nn.h)
    fwrite(&nn->num_layers, sizeof(int), 1, f);

    for(int l = 0; l < nn->num_layers; l++) {
        Linear* layer = &nn->layers[l];
        Tensor* W = layer->weight;
        Tensor* b = layer->bias;

        int64_t w0 = W->shape[0], w1 = W->shape[1];
        int64_t b0 = b->shape[0], b1 = b->shape[1];

        fwrite(&w0, sizeof(int64_t), 1, f);
        fwrite(&w1, sizeof(int64_t), 1, f);
        fwrite(W->data, sizeof(float), (size_t)(w0*w1), f);

        fwrite(&b0, sizeof(int64_t), 1, f);
        fwrite(&b1, sizeof(int64_t), 1, f);
        fwrite(b->data, sizeof(float), (size_t)(b0*b1), f);
    }

    fclose(f);
}

static void load_model(const char* file_path, const MLP* nn) {
    FILE* f = fopen(file_path, "rb");
    if(!f) {
        fatal("save_model: failed to open %s", file_path);
    }

    // UPDATE THE METADATA READ/LOADED IF NOT UPDATED
    fread(&nn->num_layers, sizeof(int), 1, f);

    for(int l = 0; l < nn->num_layers; l++) {
        Linear* layer = &nn->layers[l];
        Tensor* W = layer->weight;
        Tensor* b = layer->bias;

        int64_t w0 = W->shape[0], w1 = W->shape[1];
        int64_t b0 = b->shape[0], b1 = b->shape[1];

        fread(&w0, sizeof(int64_t), 1, f);
        fread(&w1, sizeof(int64_t), 1, f);
        fread(W->data, sizeof(float), (size_t)(w0*w1), f);

        fread(&b0, sizeof(int64_t), 1, f);
        fread(&b1, sizeof(int64_t), 1, f);
        fread(b->data, sizeof(float), (size_t)(b0*b1), f);
    }

    fclose(f);

}

#endif