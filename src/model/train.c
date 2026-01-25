#include "nn.h"
#include "model.h"
#include "dataset.h"
#include "optim.h"
#include "loss.h"
#include "graph.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <errno.h>

static int parse_int(const char *s, const char *name) {
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);

    if(errno || end == s || *end != '\0') {
        fprintf(stderr, "Invalid integer for %s: '%s'\n", name, s);
        exit(2);
    }

    return (int) v;
}

static float parse_float(const char *s, const char *name) {
    char *end = NULL;
    errno = 0;
    float v = strtof(s, &end);

    if(errno || end == s || *end != '\0') {
        fprintf(stderr, "Invalid float for %s: '%s'\n", name, s);
        exit(2);
    }

    return v;
}

// #var creates var as a string
#define SET_INT(var) do { (var) = parse_int(optarg, #var); } while(0)
#define SET_FLOAT(var) do { (var) = parse_float(optarg, #var); } while(0)


static struct option long_opts[] = {
    {"help", no_argument, 0, 'h'},
    {"d_shape", required_argument, 0, 'd'},
    {"num_classes", required_argument, 0, 'n'},
    {"num_data", required_argument, 0, 'p'},
    {"rotations", required_argument, 0, 'r'},
    {"nstd", required_argument, 0, 'j'},
    {"layers", required_argument, 0, 'l'},
    {"inputdim", required_argument, 0, 'k'},
    {"width", required_argument, 0, 'w'},
    {"outputdim", required_argument, 0, 'z'},
    {"epochs", required_argument, 0, 'e'},
    {"lr", required_argument, 0, 't'},
    {0, 0, 0, 0}
};

// TODO: maybe wrap this into a main .c file, where inference and training can be toggled.
// inference must have load flag set though
int main(int argc, char* argv[]) {
    int opt = 0;

    char* input_file = NULL;
    char* output_file = NULL;

    int data_shape = 0;
    int num_classes = 2;
    int n_per_class = 600;
    float rotations = 3.0f;
    float noise_std = 0.03f;

    int num_layers = 10;
    int input_dim = 2; // code now uses only the harcoded variant
    int width = 20;
    int output_dim = 2;

    int training_epochs = 100;
    float lr = 0.03f;

    // Hardcoded for now
    Activation hidden_activation = ACT_SOFTMAX;
    InitScheme hidden_init = INIT_HE_NORMAL;
    InitScheme output_init = INIT_HE_NORMAL;
    int hard_coded_input_dim = 2;

    // MOre of seed than rng
    uint32_t rng = 12345;

    // TODO: registry for datasetshape, so user can flag into the right dataset shape (add flag);
    // Flags might blow up when we add more dataset shapes........ hmmm.....
    const char help_menu = "\nUsage: %s [options/flags]\n"
                        "===================== Options/Flags =====================\n"
                        "-m                                Mute this error message\n"
                        "-i <file_path>                       Load model from path\n"
                        "-o <file_path>                         Save model to path\n"
                        "-d_shape <int>                              Dataset shape\n"
                        "-num_classes <int>                # of classes in dataset\n"
                        "-num_data <int>                # of data points per class\n"
                        "-rotations <float>        # of Rotations for spirals data\n"
                        "-nstd <float>                std of noise in spirals data\n"
                        "-layers <int>                             # Layers in MLP\n"
                        "-inputdim <int>                       # of dims for input\n"
                        "-width <int>                   # of dims for hidden layer\n"
                        "-outputdim <int>                     # of dims for output\n"
                        "-epochs <int>                        # of training epochs\n"
                        "-lr <float>                        Learning rate of model\n";

    // When no arguments are provided by the user at all (min value for argc is 1), the help menu
    // for flags comes up
    if(argc == 1) {
        fprintf(stderr, help_menu, argv[0]);

        return 1;
    }

    /* - getopt_long(int argc, char *const argv[], const char *optstring, const struct option *longopts, int *longindex)
       - optstring is just a string of chars, each char representing a char option
       - ':' in optstring indicates the immediate prefix char needs an argument
        - eg optstring: "a:b" -a requires an argument (eg -a value or -a=value) 
        which will be available in the external variable optarg. -b is a standalone flag 
        and does not take an argument.
        - eg could be --epoch 3 or --epoch=3 as well
        - leading colon however eg optstring: ":a:b" enables 
        "silent error reporting mode" (also known as POSIX-style error handling).
       - longopts is a struct for the longer option to single char conversion
       - If non-NULL, *longindex will be set to the index in longopts[] of the matched option, most people pass NULL.
    */
    while((opt = getopt_long(argc, argv, "hmi:o:d:n:p:r:j:l:k:w:z:e:t:", long_opts, NULL)) != -1) {
        switch(opt) {
            case 'h':
                fprintf(stderr, help_menu, argv[0]);
                return 0;
            case 'm':
                break;
            case 'i': input_file = optarg; break;
            case 'o': output_file = optarg; break;
            case 'd': SET_INT(data_shape); break;
            case 'n': SET_INT(num_classes); break;
            case 'p': SET_INT(n_per_class); break;
            case 'r': SET_FLOAT(rotations); break;
            case 'j': SET_FLOAT(noise_std); break;
            case 'l': SET_INT(num_layers); break;
            case 'k': SET_INT(input_dim); break;
            case 'w': SET_INT(width); break;
            case 'z': SET_INT(output_dim); break;
            case 'e': SET_INT(training_epochs); break;
            case 't': SET_FLOAT(lr); break;
            default:
                fprintf(stderr, "INVALID FLAG/ARGUMENT");
                fprintf(stderr, help_menu, argv[0]);
                return 1;
        }
    }

    Arena param_arena;
    // 1 mb
    arena_init(&param_arena, 1 << 20);
    
    // Scratch aren for each epoch
    Arena scratch;
    arena_init(&scratch, 1 << 20);

    Arena data_arena;
    Dataset dataset;
    arena_init(&data_arena, 1 << 20);

    // DatasetShape is hardcoded for now
    generate_dataset(&dataset, &data_arena, hard_coded_input_dim, n_per_class, num_classes, DATA_SPIRAL, rng);
    int* shuffle_dpoint_arr = (int*) malloc((size_t) n_per_class * sizeof(int));
    int* shuffle_class_arr = (int*) malloc((size_t) num_classes * sizeof(int));

    if(!shuffle_dpoint_arr) {
        fatal("malloc for shuffle_dpoint_arr failed");
    }
    for(int i = 0; i < n_per_class; i++){
        shuffle_dpoint_arr[i] = i;
    }
    if(!shuffle_class_arr) {
        fatal("malloc for shuffle_class_arr failed");
    }
    for(int i = 0; i < num_classes; i++) {
        shuffle_class_arr[i] = i;
    }


    MLP nn;
    // TODO: shouldnt be hardcodedinput dim for the input dim, its meant to be for the wdith
    init_mlp(&nn, &param_arena, num_layers, hard_coded_input_dim, width,
        output_dim, hidden_activation, hidden_init, output_init, &rng);

    // add a shuffle for the class dimension, 
    for(int epoch = 1; epoch <= training_epochs; epoch++) {
        shuffle_indexes(shuffle_dpoint_arr, n_per_class, rng);
        shuffle_indexes(shuffle_class_arr,  num_classes, rng);

        float loss_sum = 0.0f;
        int correct = 0;

        // BGD for now
        // Not sure if this shuffling is sufficient, loop per data isntance outside then per class inside.
        // TODO: might not need the additional dimension for the storage of the class since we already have the value at the start of the training loop
       for(int it = 0; it < n_per_class; it++) {
            int data_idx = shuffle_dpoint_arr[it];

            for(int j = 0; j < num_classes; j++) {
                int class_idx = shuffle_class_arr[j];

                arena_reset(&scratch);

                Graph graph;
                graph_init(&graph, &scratch);

                // Value 3 is hard_coded_input_dim + 1 for class index
                int64_t input_shape[2] = { 1, hard_coded_input_dim };
                Tensor* tensor = tensor_new(&scratch, hard_coded_input_dim, input_shape);
                size_t num_elements = total_elems(tensor);

                // TODO: Pseudo rowmajor strides calc as well
                size_t base_idx = n_per_class * (class_idx * num_classes + data_idx);

                for(int k = 0; k < num_elements; k++) {
                    tensor->data[k] = dataset.class_dpoints[base_idx + k];
                }

                Node* input_node = graph_add_input(&graph, tensor);
                Node* output_node = mlp_forward(&graph, input_node, &nn);
                Node* final_output = apply_activation(&graph, ACT_SOFTMAX, output_node);

                Node** order = NULL;
                size_t order_n = 0;
                topo_sort(&graph, &order, &order_n);
                forward(&graph, order, order_n);

                // TODO: Fix dims for the probs output
                float loss = cross_entropy(final_output->out->data, class_idx);
                loss_sum += loss;

                int pred = 0;

                for(size_t i = 0; i < final_output->out->ndim; i++) {
                    pred = final_output->out->data[pred] > final_output->out->data[i]? final_output->out->data[pred] : final_output->out->data[i];
                }

                if(pred == class_idx) {
                    correct++;
                }

                ensure_grad(&scratch, output_node->out);

                // TODO: Check
                for(size_t i = 0; i < output_node->out->ndim; i++) {
                    output_node->out->grad->data[i] = final_output->out->data[i] - (i == class_idx? 1.0f : 0.0f);
                }

                graph_backward_pass(&graph, order, order_n, output_node->out->grad);
                // TODO: Implement per op, pseudo for now
                // mlp_sgd_step(&nn, lr);

                mlp_zero_grads(&nn);

                free(order);
                graph_free(&graph);
            }
        }
        float avg_loss = loss_sum / (float) n_per_class;
        float acc = (float)correct / (float) n_per_class;

        if(epoch % 10 == 0 || epoch == 1 || epoch == training_epochs) {
            printf("Epoch %4d | loss %.6f | acc %.3f\n", epoch, avg_loss, acc);
        }
    }

    // TODO: Pseudo for now
    // float final_acc = eval_accuracy(&nn, &dataset);
    // printf("Final accuracy (train set): %.3f\n", final_acc);

    save_model("spirals_model.bin", &nn);
    printf("Saved model to spirals_model.bin\n");

    mlp_free(&nn);
    arena_free(&param_arena);
    arena_free(&scratch);
    free(shuffle_class_arr);
    free(shuffle_dpoint_arr);
    free_dataset(&dataset);

    return 0;
}