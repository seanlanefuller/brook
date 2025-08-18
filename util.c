#include "brook.h"

void to_lowercase(char* s) {
    for (int i = 0; s[i]; i++)
        s[i] = tolower((unsigned char)s[i]);
}

// Assumes row-major W[out_size][in_size]
// x and out are aligned float arrays
void fast_matmul(const float * restrict W,
                 const float * restrict x,
                 float * restrict out,
                 int out_size,
                 int in_size)
{
    // Pick tile size for cache line efficiency
    const int TILE_SIZE = 64;

    // Initialize output
    for (int i = 0; i < out_size; i++) {
        out[i] = 0.0f;
    }

    // Main tiled loop
    for (int jj = 0; jj < in_size; jj += TILE_SIZE) {
        int j_end = (jj + TILE_SIZE < in_size) ? (jj + TILE_SIZE) : in_size;

        for (int i = 0; i < out_size; i++) {
            const float *w_ptr = &W[i * in_size + jj];
            const float *x_ptr = &x[jj];
            float sum = 0.0f;

            int j = 0;
            // Unroll by 8 for better ILP (Instruction Level Parallelism)
            for (; j + 7 < (j_end - jj); j += 8) {
                sum += w_ptr[j]     * x_ptr[j] +
                       w_ptr[j + 1] * x_ptr[j + 1] +
                       w_ptr[j + 2] * x_ptr[j + 2] +
                       w_ptr[j + 3] * x_ptr[j + 3] +
                       w_ptr[j + 4] * x_ptr[j + 4] +
                       w_ptr[j + 5] * x_ptr[j + 5] +
                       w_ptr[j + 6] * x_ptr[j + 6] +
                       w_ptr[j + 7] * x_ptr[j + 7];
            }
            // Handle remainder
            for (; j < (j_end - jj); j++) {
                sum += w_ptr[j] * x_ptr[j];
            }

            out[i] += sum;
        }
    }
}



