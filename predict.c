// Optimized predict with persistent buffers, top-k softmax, and safe numerics.
// Usage:
//   predict_init();                // once after model/weights/vars are loaded
//   int tok = predict(context, n); // many times
//   predict_cleanup();             // once at shutdown

#include "brook.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

static float *predict_x = NULL;               // MAX_EMBED
static float **predict_h = NULL;              // per-layer activations
static float *predict_logits = NULL;          // OUTPUT_SIZE
static int *predict_top_idx = NULL;           // top_k indices
static float *predict_top_val = NULL;         // top_k logits (pre-softmax exp values)
static int predict_allocated = 0;

void predict_init()
{
    if (predict_allocated) return;

    // Allocate once and reuse
    predict_x = malloc(MAX_EMBED * sizeof(float));

    predict_h = malloc(num_hidden_layers * sizeof(float *));
    for (int i = 0; i < num_hidden_layers; ++i) {
        predict_h[i] = malloc(hidden_sizes[i] * sizeof(float));
    }

    predict_logits = malloc(OUTPUT_SIZE * sizeof(float));

    // top-k buffers sized to a safe upper bound (choose 32 if you want, but we pick 64)
    // We'll allow dynamic top_k at runtime but allocate max possible = vocab_size (worst-case)
    // For memory efficiency you may prefer a fixed upper bound like 256.
    int max_topk = (vocab_size < 256) ? vocab_size : 256;
    predict_top_idx = malloc(max_topk * sizeof(int));
    predict_top_val = malloc(max_topk * sizeof(float));

    // Seed RNG once. If you want reproducible output, call srand(...) yourself BEFORE predict_init.
    srand((unsigned)time(NULL));

    predict_allocated = 1;
}

void predict_cleanup()
{
    if (!predict_allocated) return;

    free(predict_x);
    for (int i = 0; i < num_hidden_layers; ++i) free(predict_h[i]);
    free(predict_h);
    free(predict_logits);
    free(predict_top_idx);
    free(predict_top_val);

    predict_x = NULL;
    predict_h = NULL;
    predict_logits = NULL;
    predict_top_idx = NULL;
    predict_top_val = NULL;
    predict_allocated = 0;
}

int predict(int* context, int context_len)
{
    if (!predict_allocated) {
        // Auto-init if user forgot (optional)
        predict_init();
    }

    if (context_len <= 0) return 0;

    // effective context (match training)
    int effective_context = context_len > context_window ? context_window : context_len;
    if (effective_context > MAX_CONTEXT) effective_context = MAX_CONTEXT;
    if (effective_context <= 0) return 0;

    // Build input vector x: zero then accumulate embeddings + positional
    for (int j = 0; j < MAX_EMBED; ++j) predict_x[j] = 0.0f;

    for (int i = 0; i < effective_context; ++i) {
        int id = context[i];
        if (id < 0 || id >= vocab_size) continue; // skip invalid
        float pos_w = 1.0f - ((float)i / (float)effective_context) * (float)POSITIONAL_DECAY_RATE;
        if (pos_w < 0.0f) pos_w = 0.0f; // clamp to avoid negative weighting (match training if needed)
        float *emb = embed[id];
        float *pos = pos_embed[i % MAX_CONTEXT];
        for (int j = 0; j < MAX_EMBED; ++j) {
            predict_x[j] += pos_w * (emb[j] + pos[j]);
        }
    }

    // Forward through hidden layers (no dropout)
    float *h_prev = predict_x;
    for (int layer = 0; layer < num_hidden_layers; ++layer) {
        int current_size = hidden_sizes[layer];
        int input_size = (layer == 0) ? MAX_EMBED : hidden_sizes[layer - 1];

        fast_matmul(W[layer], h_prev, predict_h[layer], current_size, input_size);
        // relu without dropout - train_flag = 0
        relu_and_dropout_combined(predict_h[layer], current_size, DROPOUT_RATE, 0);
        h_prev = predict_h[layer];
    }

    // Output logits (zero and matmul)
    int final_layer_size = hidden_sizes[num_hidden_layers - 1];
    int max_consider = (vocab_size < OUTPUT_SIZE) ? vocab_size : OUTPUT_SIZE;
    for (int i = 0; i < OUTPUT_SIZE; ++i) predict_logits[i] = 0.0f;
    fast_matmul(W_output, h_prev, predict_logits, OUTPUT_SIZE, final_layer_size);

    // Find global max_logit across the vocab (for numerical stability)
    float max_logit = -FLT_MAX;
    for (int i = 0; i < max_consider; ++i) {
        if (predict_logits[i] > max_logit) max_logit = predict_logits[i];
    }

    // Decide top_k (you previously used 5). Keep same behavior but allow <= vocab_size.
    int top_k = 5;
    if (top_k > max_consider) top_k = max_consider;

    // Initialize top_k arrays with the first top_k logits (we'll track the minimum)
    int filled = 0;
    float cur_min = FLT_MAX;
    int cur_min_idx = -1;
    for (int i = 0; i < max_consider; ++i) {
        float val = predict_logits[i];
        if (filled < top_k) {
            predict_top_idx[filled] = i;
            predict_top_val[filled] = val;
            filled++;
            if (val < cur_min) { cur_min = val; cur_min_idx = filled - 1; }
            if (filled == top_k) {
                // ensure we have correct cur_min/cur_min_idx
                cur_min = predict_top_val[0];
                cur_min_idx = 0;
                for (int k = 1; k < top_k; ++k) {
                    if (predict_top_val[k] < cur_min) { cur_min = predict_top_val[k]; cur_min_idx = k; }
                }
            }
        } else {
            // If current logit > current minimum in top_k, replace it and update min
            if (val > cur_min) {
                predict_top_idx[cur_min_idx] = i;
                predict_top_val[cur_min_idx] = val;
                // recompute min and index (cost = O(top_k) but top_k is small)
                cur_min = predict_top_val[0];
                cur_min_idx = 0;
                for (int k = 1; k < top_k; ++k) {
                    if (predict_top_val[k] < cur_min) { cur_min = predict_top_val[k]; cur_min_idx = k; }
                }
            }
        }
    }

    // Compute (stable) softmax over top_k only: exp((logit - max_logit)/T) with clamping
    float sum_exp = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        // Use the global max_logit for stability.
        float z = (predict_top_val[k] - max_logit) / (TEMPERATURE <= 0.0f ? 1e-6f : TEMPERATURE);

        // clamp z to avoid overflow/underflow
        if (z > 50.0f) z = 50.0f;
        if (z < -50.0f) z = -50.0f;

        // store the exp in-place (reuse predict_top_val to hold exp values now)
        float e = expf(z);
        predict_top_val[k] = e;
        sum_exp += e;
    }

    // normalize (safeguard if sum_exp == 0)
    if (!(sum_exp > 0.0f)) sum_exp = 1e-12f;
    for (int k = 0; k < top_k; ++k) predict_top_val[k] /= sum_exp;

    // Sample from the top-k distribution
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    int selected = predict_top_idx[0];
    for (int k = 0; k < top_k; ++k) {
        cumsum += predict_top_val[k];
        if (r <= cumsum) { selected = predict_top_idx[k]; break; }
    }

    return selected;
}
