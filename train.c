#include "brook.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function to perform He initialization for weights
// This is a crucial step to prevent vanishing gradients with ReLU activations
void he_init(float* W, int fan_in, int fan_out) {
    // Standard deviation is sqrt(2 / fan_in) - but make it smaller for stability
    float std_dev = sqrtf(2.0f / fan_in);  // Reduce from 2.0f to 1.0f
    // Fill the weight matrix with random values from a normal distribution
    // This is a simple approximation using a uniform distribution
    for (int i = 0; i < fan_in * fan_out; i++) {
        W[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * std_dev;
    }
}

int first_time = 1;
float **dW = NULL;
float **h_activations = NULL;
float *x_input_buffer;
float* dW_output = NULL;
float** deltas = NULL;
float* output_deltas = NULL;
float initial_lr = 0.0f;
int effective_context = 0;
int prev_size = MAX_EMBED;
float* logits = NULL;
float* h_prev = NULL;
float* probs = NULL;
int target = 0;
float current_lr = 0;
int final_layer_size = 0;

void init_training(int max_context)
{
    srand((unsigned int)time(NULL));

    // Allocate memory for gradients.
    dW = malloc(num_hidden_layers * sizeof(float*));
    for (int i = 0; i < num_hidden_layers; i++) {
        dW[i] = calloc(hidden_sizes[i] * ((i == 0) ? MAX_EMBED : hidden_sizes[i - 1]), sizeof(float));
    }

	dW_output = calloc(OUTPUT_SIZE * hidden_sizes[num_hidden_layers - 1], sizeof(float));

    // Allocate memory for hidden layer activations.
    h_activations = malloc(num_hidden_layers * sizeof(float*));
    for (int i = 0; i < num_hidden_layers; i++) {
        h_activations[i] = malloc(hidden_sizes[i] * sizeof(float));
    }
    x_input_buffer = malloc(MAX_EMBED * sizeof(float));
    
    // A buffer to store the deltas (error signals) for each layer
    deltas = malloc(num_hidden_layers * sizeof(float*));
    for (int i = 0; i < num_hidden_layers; i++) {
        deltas[i] = malloc(hidden_sizes[i] * sizeof(float));
    }
    output_deltas = malloc(OUTPUT_SIZE * sizeof(float));

	probs = malloc(vocab_size * sizeof(float));
	logits = malloc(OUTPUT_SIZE * sizeof(float));

    if (first_time && !get_loaded_weights()) {
        // Initialize weights using He initialization
        for (int i = 0; i < num_hidden_layers; i++) {
            int fan_in = (i == 0) ? MAX_EMBED : hidden_sizes[i - 1];
            int fan_out = hidden_sizes[i];
            he_init(W[i], fan_in, fan_out);
        }
        he_init(W_output, hidden_sizes[num_hidden_layers - 1], OUTPUT_SIZE);
        first_time = 0;
    }	
    initial_lr = LEARNING_RATE;
    effective_context = max_context > context_window ? context_window : max_context;
}

void forward_pass(int i)
{
	memset(x_input_buffer, 0, MAX_EMBED * sizeof(float));
	for (int p = 0; p < effective_context; p++) {
		int id = tokens[i + p];
		if (id < 0 || id >= vocab_size) {
			printf("Error: Invalid token ID %d at position %d (vocab_size=%d)\n", id, i+p, vocab_size);
			return;
		}
		float weight = 1.0f - (float)p / (float)effective_context * (float)POSITIONAL_DECAY_RATE;
		for (int j = 0; j < MAX_EMBED; j++) {
			x_input_buffer[j] += weight * (embed[id][j] + pos_embed[p % MAX_CONTEXT][j]);
		}
	}
	
	// Print input statistics
	if (i % 100 == 0) {
		float input_sum = 0, input_max = -1e9, input_min = 1e9;
		for (int j = 0; j < MAX_EMBED; j++) {
			input_sum += x_input_buffer[j];
			if (x_input_buffer[j] > input_max) input_max = x_input_buffer[j];
			if (x_input_buffer[j] < input_min) input_min = x_input_buffer[j];
		}
		if (DEBUG) {
			printf("  Input[%d]: avg=%.4f, min=%.4f, max=%.4f\n", 
				i, input_sum/MAX_EMBED, input_min, input_max);
		}
	}

	// Hidden layers forward pass
	h_prev = x_input_buffer;
	for (int layer = 0; layer < num_hidden_layers; layer++) {
		int current_size = hidden_sizes[layer];
		int input_size = (layer == 0) ? MAX_EMBED : hidden_sizes[layer - 1];
		fast_matmul(W[layer], h_prev, h_activations[layer], current_size, input_size);
		
		// Check pre-activation values
		if (i % 100 == 0) {
			float pre_act_sum = 0, pre_act_max = -1e9, pre_act_min = 1e9;
			for (int j = 0; j < current_size; j++) {
				pre_act_sum += h_activations[layer][j];
				if (h_activations[layer][j] > pre_act_max) pre_act_max = h_activations[layer][j];
				if (h_activations[layer][j] < pre_act_min) pre_act_min = h_activations[layer][j];
			}
			if (DEBUG) {
				printf("  Layer%d pre-ReLU[%d]: avg=%.4f, min=%.4f, max=%.4f\n", 
					layer, i, pre_act_sum/current_size, pre_act_min, pre_act_max);
			}
		}
		
		relu_and_dropout_combined(h_activations[layer], current_size, DROPOUT_RATE, 1);

		// Check post-activation values
		if (i % 100 == 0) {
			float post_act_sum = 0, zeros = 0;
			for (int j = 0; j < current_size; j++) {
				post_act_sum += h_activations[layer][j];
				if (h_activations[layer][j] == 0) zeros++;
			}
			if (DEBUG) {
				printf("  Layer%d post-ReLU[%d]: avg=%.4f, zeros=%d/%d (%.1f%%)\n", 
				   layer, i, post_act_sum/current_size, (int)zeros, current_size, 100.0f*zeros/current_size);
				}
		}

		// Set up the next layer's input
		h_prev = h_activations[layer];
		prev_size = current_size;
	}

	// Output layer forward pass
	memset(logits, 0, OUTPUT_SIZE * sizeof(float));
	fast_matmul(W_output, h_prev, logits, OUTPUT_SIZE, prev_size);
	
	// Check output logits
	if (i % 100 == 0) {
		float logit_sum = 0, logit_max = -1e9, logit_min = 1e9;
		for (int j = 0; j < vocab_size; j++) {
			logit_sum += logits[j];
			if (logits[j] > logit_max) logit_max = logits[j];
			if (logits[j] < logit_min) logit_min = logits[j];
		}
		if (DEBUG) {
			printf("  Logits[%d]: avg=%.4f, min=%.4f, max=%.4f\n", 
				i, logit_sum/vocab_size, logit_min, logit_max);
		}
	}
}

void backward_pass()
{
	// Calculate output layer deltas (error signal)
	for (int j = 0; j < OUTPUT_SIZE; j++) {
		if (j < vocab_size) {
			output_deltas[j] = probs[j] - (j == target ? 1.0f : 0.0f);
		} else {
			output_deltas[j] = 0.0f;  // No gradient for unused outputs
		}
	}

	// Update output weights gradients
	for (int j = 0; j < OUTPUT_SIZE; j++) {
		for (int k = 0; k < prev_size; k++) {
			// Accumulate gradient
			dW_output[j * prev_size + k] += output_deltas[j] * h_activations[num_hidden_layers - 1][k];
		}
	}
	
	// Backpropagate through hidden layers
	float* next_deltas = output_deltas;
	int next_size = OUTPUT_SIZE;  // Use OUTPUT_SIZE for consistency
	float* next_weights = W_output;
	int next_input_size = hidden_sizes[num_hidden_layers - 1];
	
	for (int layer = num_hidden_layers - 1; layer >= 0; layer--) {
		float* h_current = (layer == 0) ? x_input_buffer : h_activations[layer - 1];
		int h_current_size = (layer == 0) ? MAX_EMBED : hidden_sizes[layer - 1];
		
		// Calculate deltas for current layer
		for (int j = 0; j < hidden_sizes[layer]; j++) {
			float sum_err = 0.0f;
			for (int k = 0; k < next_size; k++) {
				sum_err += next_weights[k * next_input_size + j] * next_deltas[k];
			}
			// Derivative of ReLU (chain rule)
			// This is where a dead ReLU neuron will have a delta of 0
			if (h_activations[layer][j] > 0) {
				deltas[layer][j] = sum_err;
			} else {
				deltas[layer][j] = 0;
			}
		}
		
		// Update gradients for current layer
		for (int j = 0; j < hidden_sizes[layer]; j++) {
			for (int k = 0; k < h_current_size; k++) {
				// Accumulate gradient
				dW[layer][j * h_current_size + k] += deltas[layer][j] * h_current[k];
			}
		}

		// Prepare for next layer (going backwards)
		next_deltas = deltas[layer];
		next_size = hidden_sizes[layer];
		next_weights = W[layer];
		next_input_size = h_current_size;
	}
}

void update_weights()
{
	// ---- WEIGHT UPDATE PASS (AFTER ALL BATCH GRADIENTS ARE CALCULATED) ----
	// This is a more stable approach than updating every iteration
	// Update output weights - fix dimension mismatch
	final_layer_size = hidden_sizes[num_hidden_layers - 1];
	
	// Track gradient statistics
	float grad_sum = 0, grad_max = -1e9, grad_min = 1e9;
	int grad_count = 0;
	
	for (int j = 0; j < OUTPUT_SIZE; j++) {  // Use OUTPUT_SIZE, not vocab_size
		for (int k = 0; k < final_layer_size; k++) {
			if (j < vocab_size) {  // Only update weights for actual vocabulary
				float grad = dW_output[j * final_layer_size + k];  // Remove division - raw accumulated gradient
				
				grad_sum += fabsf(grad);
				if (grad > grad_max) grad_max = grad;
				if (grad < grad_min) grad_min = grad;
				grad_count++;

				// Gradient clipping - slightly looser for faster learning
				if (grad > 0.5f) grad = 0.5f;   // Increase from 0.1f to 0.5f
				if (grad < -0.5f) grad = -0.5f;
				W_output[j * final_layer_size + k] -= current_lr * grad;
			}
		}
	}
	
	if (DEBUG) {
		printf("  Output grads: avg=%.6f, min=%.6f, max=%.6f, lr=%.6f\n", 
			grad_sum/grad_count, grad_min, grad_max, current_lr);
	}
	
	// Update hidden layer weights
	for (int layer = 0; layer < num_hidden_layers; layer++) {
		int current_size = hidden_sizes[layer];
		int input_size = (layer == 0) ? MAX_EMBED : hidden_sizes[layer - 1];
		
		float hidden_grad_sum = 0;
		int hidden_grad_count = 0;
		
		for (int j = 0; j < current_size; j++) {
			for (int k = 0; k < input_size; k++) {
				float grad = dW[layer][j * input_size + k];  // Remove division - raw accumulated gradient
				hidden_grad_sum += fabsf(grad);
				hidden_grad_count++;
				
				// Gradient clipping - slightly looser for faster learning
				if (grad > 0.5f) {
					grad = 0.5f;   // Increase from 0.1f to 0.5f
				}
				else if (grad < -0.5f) {
					grad = -0.5f;
				}
				W[layer][j * input_size + k] -= current_lr * grad;
			}
		}
		if (DEBUG) {
			printf("  Layer%d grads: avg=%.6f\n", layer, hidden_grad_sum/hidden_grad_count);
		}
	}
}

void softmax()
{
	// Softmax - ensure we don't exceed either array bounds
	int max_vocab_idx = (vocab_size < OUTPUT_SIZE) ? vocab_size : OUTPUT_SIZE;
	
	float max_logit = logits[0];
	for (int j = 1; j < max_vocab_idx; j++) {
		if (logits[j] > max_logit) max_logit = logits[j];
	}
	float sum_exp = 0;
	for (int j = 0; j < vocab_size; j++) {
		if (j < OUTPUT_SIZE) {
			probs[j] = expf(logits[j] - max_logit);
		} else {
			probs[j] = 0.0f;  // Should never happen if sizes are correct
		}
		sum_exp += probs[j];
	}
	
	// Check for numerical issues
	if (sum_exp == 0 || !isfinite(sum_exp)) {
		printf("Error: sum_exp=%f, max_logit=%f\n", sum_exp, max_logit);
		sum_exp = 1e-8f;  // Prevent division by zero
	}
	
	for (int j = 0; j < vocab_size; j++) {
		probs[j] /= sum_exp;
	}
}

void clear_gradients()
{
	// Clear gradients for the next epoch
	for (int i = 0; i < num_hidden_layers; i++) {
		memset(dW[i], 0, hidden_sizes[i] * ((i == 0) ? MAX_EMBED : hidden_sizes[i - 1]) * sizeof(float));
	}
	int output_layer_size = hidden_sizes[num_hidden_layers - 1];
	memset(dW_output, 0, OUTPUT_SIZE * output_layer_size * sizeof(float));
}

void training_cleanup()
{
    // Free allocated memory
    for (int i = 0; i < num_hidden_layers; i++) {
        free(dW[i]);
        free(h_activations[i]);
        free(deltas[i]);
    }
    free(dW);
    free(dW_output);
    free(h_activations);
    free(x_input_buffer);
    free(deltas);
    free(output_deltas);
	free(probs);
	free(logits);
}

void report_progress(int training_epoch, float total_loss, time_t epoch_start)
{
	// Progress reporting
	if (training_epoch % 5 == 0 || training_epoch < 20) {
		float avg_loss = total_loss / (float)(token_count - effective_context - 1);
		
		// Get current timestamp and calculate epoch duration
		time_t now = time(NULL);
		struct tm *t = localtime(&now);
		double epoch_duration = difftime(now, epoch_start);
		
		printf("[%02d:%02d:%02d] Epoch %05d Loss: %.4f Avg Loss: %.4f LR: %.6f Time: %.1fs\n",
				t->tm_hour, t->tm_min, t->tm_sec, training_epoch, total_loss, avg_loss, current_lr, epoch_duration);
				
		// Print weight statistics every 20 epochs
		if (training_epoch % 20 == 0) {
			float w_sum = 0, w_max = -1e9, w_min = 1e9;
			int w_count = 0;
			for (int j = 0; j < vocab_size && j < OUTPUT_SIZE; j++) {
				for (int k = 0; k < final_layer_size; k++) {
					float w = W_output[j * final_layer_size + k];
					w_sum += fabsf(w);
					if (w > w_max) w_max = w;
					if (w < w_min) w_min = w;
					w_count++;
				}
			}
			if (DEBUG) {
				printf("  Output weights: avg=%.4f, min=%.4f, max=%.4f\n", 
						w_sum/w_count, w_min, w_max);
			}
		}
	}
	// Print progress report
	if (DEBUG) {
		printf("Progress: Tokens processed: %d, Effective context: %d, Current learning rate: %.6f\n", 
			token_count, effective_context, current_lr);
	}
}

// Function to perform the training loop with backpropagation
void train(int max_context, int epochs) {
	init_training(max_context);

    printf("Vocab size: %d, Token count: %d, Effective context: %d\n", 
           vocab_size, token_count, effective_context);
    printf("Hidden layers: %d, Hidden sizes: %d, %d, %d\n", num_hidden_layers, 
		hidden_sizes[0], hidden_sizes[1], hidden_sizes[2]);
    printf("Training samples: %d\n", token_count - effective_context - 1);
	printf("Initial learning rate: %.6f, Context window: %d\n", 
		   initial_lr, context_window);

    for (int training_epoch = 0; training_epoch < epochs; training_epoch++) {
        time_t epoch_start = time(NULL);

        // Calculate the current learning rate with decay.
        current_lr = initial_lr * powf(DECAY_RATE, training_epoch / 10.0f);
        if (current_lr < initial_lr * 0.01f) current_lr = initial_lr * 0.01f;
        float total_loss = 0.0f;

        for (int i = 0; i < token_count - effective_context - 1; i++) {
			forward_pass(i);
			softmax();

            // Calculate loss
            target = tokens[i + effective_context];
            if (target >= 0 && target < vocab_size) {
                float sample_loss = -logf(probs[target] + 1e-8f);
                total_loss += sample_loss;
            } else {
                printf("Warning: Invalid target token %d at position %d\n", target, i + effective_context);
                total_loss += 10.0f;  // Large penalty for invalid tokens
            }
			backward_pass();
        }
		update_weights();
		clear_gradients();
		report_progress(training_epoch, total_loss, epoch_start);

		if ((training_epoch + 1) % 10 == 0 && training_epoch > 0) {
			save_model();
		}
		
    }
    training_cleanup();
}
