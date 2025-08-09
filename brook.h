#pragma once
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <ctype.h>

// Model architecture constants
#define MAX_VOCAB 5100
#define MAX_VOCAB_WORD_LEN 16
#define MAX_TOKENS 64000
#define MAX_EMBED 32
#define MAX_HIDDEN_LAYERS 5
#define OUTPUT_SIZE MAX_VOCAB  // Match the actual vocabulary size
#define CONTEXT_SIZE 8
#define MIN_CONTEXT 1
#define MAX_CONTEXT 8
#define MAX_FILE_SIZE 300000
#define MAX_EPOCHS 10000

#define DEBUG 0

// Training parameters
#define LEARNING_RATE 0.003f
#define DECAY_RATE 0.996f     // Slower decay
#define EPOCHS 10
#define TEMPERATURE 1.2f      // Increase from 1.0f to add diversity
#define DROPOUT_RATE 0.0001f     // Remove dropout completely for now
#define POSITIONAL_DECAY_RATE 0.3f

// Weight Access Macros
#define W_ACCESS(layer, i, j) W[layer][(i) * ((layer == 0) ? MAX_EMBED : hidden_sizes[layer - 1]) + (j)]
#define W_OUTPUT_ACCESS(i, j) W_output[(i) * hidden_sizes[num_hidden_layers - 1] + (j)]

extern int num_hidden_layers;
extern int hidden_sizes[MAX_HIDDEN_LAYERS];
extern int context_window;
extern char vocab[MAX_VOCAB][MAX_VOCAB_WORD_LEN];
extern int vocab_size;
extern int tokens[MAX_TOKENS];
extern int token_count;
extern float learning_rate;
extern float embed[MAX_VOCAB][MAX_EMBED];
extern float pos_embed[MAX_CONTEXT][MAX_EMBED];
extern float* W[MAX_HIDDEN_LAYERS];
extern float* W_output;
extern float* activation_buffers[MAX_HIDDEN_LAYERS];
extern float* gradient_buffers[MAX_HIDDEN_LAYERS];

void he_init(float* W, int fan_in, int fan_out);
int get_loaded_weights();
void set_loaded_weights();
int get_token_id(const char* word);
void tokenize(const char* text);
void allocate_weights();
void free_weights();
void initialize_weights();
void relu_and_dropout_combined(float* v, int size, float dropout_rate, int training);
int predict(int* context, int context_len);
void train(int max_context, int epochs);
void save_model();
int load_model();
void generate_text_from_seed(const char* seed_string, int steps);
void generate_sentences(const char* seed_string, int num_sentences);
void interactive_mode();
int load_training_data(const char* filename);
void init_vocab();
void cleanup();
void to_lowercase(char* s);
int get_token_id(const char* word);
int get_token_id_existing(const char* word);
void to_lowercase(char* s);
void to_lowercase(char* s);
void tokenize_user_input(const char* text, int* out_tokens, int* out_count, int max_tokens);
void relu(float* x, int size);
void fast_matmul(const float * restrict W,
                 const float * restrict x,
                 float * restrict out,
                 int out_size,
                 int in_size);
void init_weights();
long long count_parameters();
void print_model_info();

