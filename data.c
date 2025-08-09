#include "brook.h"

// Global Configuration Variables
int num_hidden_layers = 3;  // Keep 3 layers for good representation
int hidden_sizes[MAX_HIDDEN_LAYERS] = {512, 256, 128, 64, 64};  // Pyramid structure
int context_window = CONTEXT_SIZE;
int loaded_weights = 0;

// Global Data Structures
char vocab[MAX_VOCAB][MAX_VOCAB_WORD_LEN];
int vocab_size = 0;
int tokens[MAX_TOKENS];
int token_count = 0;
float learning_rate = LEARNING_RATE;

// Neural Network Parameters
float embed[MAX_VOCAB][MAX_EMBED];
float pos_embed[MAX_CONTEXT][MAX_EMBED];
float* W[MAX_HIDDEN_LAYERS];
float* W_output;
float* activation_buffers[MAX_HIDDEN_LAYERS];
float* gradient_buffers[MAX_HIDDEN_LAYERS];

int get_loaded_weights()
{
	return loaded_weights;
}

void set_loaded_weights()
{
	loaded_weights = 1;
}

int load_training_data(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Error: Could not open %s\n", filename);
        return 0;
    }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (file_size <= 0 || file_size > MAX_FILE_SIZE) {
        printf("Error: File size %ld is invalid or too large\n", file_size);
        fclose(f);
        return 0;
    }
    char* text = (char*)malloc(file_size + 1);
    if (!text) {
        printf("Error: Could not allocate memory for training data\n");
        fclose(f);
        return 0;
    }
    size_t bytes_read = fread(text, 1, file_size, f);
    text[bytes_read] = '\0';
    fclose(f);

    printf("Loaded %zu chars from %s\n", bytes_read, filename);
    tokenize(text);
    printf("Tokenized: %d tokens, %d unique words\n", token_count, vocab_size);

    free(text);
    return token_count >= 10;
}

void init_vocab() {
    vocab_size = 0;
}

void cleanup() {
    free_weights();
}
