#include "brook.h"

void allocate_weights() {
    for (int layer = 0; layer < num_hidden_layers; layer++) {
        int input_size = (layer == 0) ? MAX_EMBED : hidden_sizes[layer - 1];
        int output_size = hidden_sizes[layer];
        W[layer] = (float*)malloc(input_size * output_size * sizeof(float));
        if (!W[layer]) {
            printf("Error: Could not allocate memory for layer %d weights\n", layer);
            exit(1);
        }
        activation_buffers[layer] = (float*)malloc(output_size * sizeof(float));
        if (!activation_buffers[layer]) {
            printf("Error: Could not allocate memory for layer %d activations\n", layer);
            exit(1);
        }
        gradient_buffers[layer] = (float*)malloc(output_size * sizeof(float));
        if (!gradient_buffers[layer]) {
            printf("Error: Could not allocate memory for layer %d gradients\n", layer);
            exit(1);
        }
    }
    int final_input_size = hidden_sizes[num_hidden_layers - 1];
    W_output = (float*)malloc(OUTPUT_SIZE * final_input_size * sizeof(float));
    if (!W_output) {
        printf("Error: Could not allocate memory for output weights\n");
        exit(1);
    }
}

void free_weights() {
    for (int layer = 0; layer < num_hidden_layers; layer++) {
        if (W[layer]) {
			free(W[layer]); W[layer] = NULL; 
		}
        if (activation_buffers[layer]) {
			free(activation_buffers[layer]); 
			activation_buffers[layer] = NULL; 
		}
        if (gradient_buffers[layer]) {
			free(gradient_buffers[layer]);
			gradient_buffers[layer] = NULL;
		}
    }
    if (W_output) { free(W_output); W_output = NULL; }
}

void initialize_weights() {
    allocate_weights();
    
    // Initialize word embeddings with Xavier initialization
    float xavier_embed = sqrtf(2.0f / (MAX_EMBED + vocab_size));
    for (int i = 0; i < MAX_VOCAB; i++) {
        for (int j = 0; j < MAX_EMBED; j++) {
            embed[i][j] = ((float)rand() / RAND_MAX - 0.5f) * xavier_embed;
        }
    }
    
    // Initialize position embeddings with small random values
    for (int i = 0; i < MAX_CONTEXT; i++) {
        for (int j = 0; j < MAX_EMBED; j++) {
            pos_embed[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        }
    }
    
    // Initialize hidden layer weights with He initialization (better for ReLU)
    for (int layer = 0; layer < num_hidden_layers; layer++) {
        int input_size = (layer == 0) ? MAX_EMBED : hidden_sizes[layer - 1];
        int output_size = hidden_sizes[layer];
        float he_scale = sqrtf(2.0f / input_size);  // He initialization for ReLU
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < input_size; j++) {
                W_ACCESS(layer, i, j) = ((float)rand() / RAND_MAX - 0.5f) * he_scale;
            }
        }
    }
    
    // Initialize output layer weights with Xavier initialization
    int final_input_size = hidden_sizes[num_hidden_layers - 1];
    float xavier_output = sqrtf(2.0f / (final_input_size + OUTPUT_SIZE));
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < final_input_size; j++) {
            W_OUTPUT_ACCESS(i, j) = ((float)rand() / RAND_MAX - 0.5f) * xavier_output;
        }
    }
}

void relu_and_dropout_combined(float* v, int size, float dropout_rate, int training) {
    if (training) {
        float inv_keep_prob = 1.0f / (1.0f - dropout_rate);
        for (int i = 0; i < size; i++) {
            // Apply ReLU activation
            if (v[i] < 0) v[i] = 0;
            
            // Apply dropout with inverse scaling
            if ((float)rand() / RAND_MAX < dropout_rate) {
                v[i] = 0;
            } else {
                v[i] *= inv_keep_prob;
            }
        }
    } else {
        // Inference mode: only apply ReLU, no dropout
        for (int i = 0; i < size; i++) {
            if (v[i] < 0) v[i] = 0;
        }
    }
}

void save_vocab() {
    FILE* f = fopen("vocab.txt", "w");
    if (!f) {
        printf("Error: Could not save vocab.txt\n");
        return;
    }
    for (int i = 0; i < vocab_size; i++) {
        fprintf(f, "%s\n", vocab[i]);
    }
    fclose(f);
}

void save_model() {
    FILE* f = fopen("weights.bin", "wb");
    if (!f) {
        printf("Error: Could not save weights.bin\n");
        return;
    }
    fwrite(&vocab_size, sizeof(int), 1, f);
    fwrite(&num_hidden_layers, sizeof(int), 1, f);
    fwrite(hidden_sizes, sizeof(int), num_hidden_layers, f);
    fwrite(embed, sizeof(embed), 1, f);
    fwrite(pos_embed, sizeof(pos_embed), 1, f);
    for (int layer = 0; layer < num_hidden_layers; layer++) {
        int input_size = (layer == 0) ? MAX_EMBED : hidden_sizes[layer - 1];
        int output_size = hidden_sizes[layer];
        fwrite(W[layer], sizeof(float), input_size * output_size, f);
    }
    int final_input_size = hidden_sizes[num_hidden_layers - 1];
    fwrite(W_output, sizeof(float), OUTPUT_SIZE * final_input_size, f);
    fclose(f);
    save_vocab();
    printf("Model saved.\n");
}

void load_vocab() {
    FILE* f = fopen("vocab.txt", "r");
    if (f) {
        vocab_size = 0;
        char line[MAX_VOCAB_WORD_LEN];
        while (fgets(line, sizeof(line), f) && vocab_size < MAX_VOCAB) {
            line[strcspn(line, "\r\n")] = 0;
            if (strlen(line) > 0) {
                size_t len = strlen(line);
                if (len >= MAX_VOCAB_WORD_LEN) len = MAX_VOCAB_WORD_LEN - 1;
                memcpy(vocab[vocab_size], line, len);
                vocab[vocab_size][len] = '\0';
                vocab_size++;
            }
        }
        fclose(f);
    }
}

int load_model() {
    FILE* f = fopen("weights.bin", "rb");
    if (!f) return 0;
    int saved_vocab_size, saved_layers, saved_sizes[MAX_HIDDEN_LAYERS];
    if (fread(&saved_vocab_size, sizeof(int), 1, f) != 1 ||
        fread(&saved_layers, sizeof(int), 1, f) != 1 ||
        fread(saved_sizes, sizeof(int), saved_layers, f) != (size_t)saved_layers) {
        printf("Error reading model configuration\n");
        fclose(f);
        return 0;
    }
    if (saved_layers > MAX_HIDDEN_LAYERS) {
        printf("Error: Saved model has %d layers, but max supported is %d\n", saved_layers, MAX_HIDDEN_LAYERS);
        fclose(f);
        return 0;
    }
    free_weights();
    num_hidden_layers = saved_layers;
    memcpy(hidden_sizes, saved_sizes, saved_layers * sizeof(int));
    allocate_weights();
    if (fread(embed, sizeof(embed), 1, f) != 1 ||
        fread(pos_embed, sizeof(pos_embed), 1, f) != 1) {
        printf("Error reading embeddings\n");
        fclose(f);
        return 0;
    }
    for (int layer = 0; layer < num_hidden_layers; layer++) {
        int input_size = (layer == 0) ? MAX_EMBED : hidden_sizes[layer - 1];
        int output_size = hidden_sizes[layer];
        if (fread(W[layer], sizeof(float), input_size * output_size, f) != (size_t)(input_size * output_size)) {
            printf("Error reading layer %d weights\n", layer);
            fclose(f);
            return 0;
        }
    }
    int final_input_size = hidden_sizes[num_hidden_layers - 1];
    if (fread(W_output, sizeof(float), OUTPUT_SIZE * final_input_size, f) != (size_t)(OUTPUT_SIZE * final_input_size)) {
        printf("Error reading output weights\n");
        fclose(f);
        return 0;
    }
    fclose(f);

    load_vocab();

    printf("Model loaded: %d layers, %d vocabulary\n", num_hidden_layers, vocab_size);
    return 1;
}

long long count_parameters() {
    long long total_params = 0;
    
    // Word embeddings: vocab_size * embedding_dimension
    total_params += (long long)vocab_size * MAX_EMBED;
    
    // Position embeddings: max_context * embedding_dimension
    total_params += (long long)MAX_CONTEXT * MAX_EMBED;
    
    // Hidden layer weights
    for (int layer = 0; layer < num_hidden_layers; layer++) {
        int input_size = (layer == 0) ? MAX_EMBED : hidden_sizes[layer - 1];
        int output_size = hidden_sizes[layer];
        total_params += (long long)input_size * output_size;
    }
    
    // Output layer weights: final_hidden_size * output_vocabulary_size
    int final_input_size = hidden_sizes[num_hidden_layers - 1];
    total_params += (long long)OUTPUT_SIZE * final_input_size;
    
    return total_params;
}

void print_model_info() {
    long long total_params = count_parameters();
    
    printf("\n=== Model Architecture ===\n");
    printf("Vocabulary size: %d\n", vocab_size);
    printf("Embedding dimension: %d\n", MAX_EMBED);
    printf("Max context length: %d\n", MAX_CONTEXT);
    printf("Number of hidden layers: %d\n", num_hidden_layers);
    
    printf("Hidden layer sizes: [");
    for (int i = 0; i < num_hidden_layers; i++) {
        printf("%d", hidden_sizes[i]);
        if (i < num_hidden_layers - 1) printf(", ");
    }
    printf("]\n");
    
    printf("Output size: %d\n", OUTPUT_SIZE);
    
    printf("\n=== Parameter Breakdown ===\n");
    printf("Word embeddings: %lld\n", (long long)vocab_size * MAX_EMBED);
    printf("Position embeddings: %lld\n", (long long)MAX_CONTEXT * MAX_EMBED);
    
    for (int layer = 0; layer < num_hidden_layers; layer++) {
        int input_size = (layer == 0) ? MAX_EMBED : hidden_sizes[layer - 1];
        int output_size = hidden_sizes[layer];
        printf("Hidden layer %d weights: %lld\n", layer + 1, (long long)input_size * output_size);
    }
    
    int final_input_size = hidden_sizes[num_hidden_layers - 1];
    printf("Output layer weights: %lld\n", (long long)OUTPUT_SIZE * final_input_size);
    
    printf("\nTotal parameters: %lld\n", total_params);
    
    // Calculate memory usage (assuming 4 bytes per float)
    double memory_mb = (double)total_params * 4.0 / (1024.0 * 1024.0);
    printf("Approximate memory usage: %.2f MB\n", memory_mb);
    printf("========================\n\n");
}

