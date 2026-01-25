#include "nn.h"
#include "model.h"
#include "dataset.h"

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
    int input_dim = 2;
    int width = 20;
    int output_dim = 2;

    int training_epochs = 100;
    float lr = 0.03f;

    // TODO: registry for datasetshape, so user can flag into the right dataset shape (add flag);
    // Flags might blow up when we add more dataset shapes........ hmmm.....
    uint32_t rng = 12345;
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

    
}