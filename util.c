#include "brook.h"

void to_lowercase(char* s) {
    for (int i = 0; s[i]; i++)
        s[i] = tolower((unsigned char)s[i]);
}

#if 0
// A helper function to safely perform matrix multiplication
// This assumes matrices are in row-major order
void fast_matmul(float* W, float* x, float* out, int out_size, int in_size) {
    // For small matrices, use simple implementation
    if (out_size <= 32 || in_size <= 32) {
        for (int i = 0; i < out_size; i++) {
            out[i] = 0.0f;
            for (int j = 0; j < in_size; j++) {
                out[i] += W[i * in_size + j] * x[j];
            }
        }
        return;
    }
    
    // Optimized version for larger matrices
    const int TILE_SIZE = 32;  // Cache-friendly tile size
    
    // Initialize output to zero
    for (int i = 0; i < out_size; i++) {
        out[i] = 0.0f;
    }
    
    // Tiled matrix multiplication with loop unrolling
    for (int ii = 0; ii < out_size; ii += TILE_SIZE) {
        for (int jj = 0; jj < in_size; jj += TILE_SIZE) {
            // Process tile
            int i_end = (ii + TILE_SIZE < out_size) ? ii + TILE_SIZE : out_size;
            int j_end = (jj + TILE_SIZE < in_size) ? jj + TILE_SIZE : in_size;
            
            for (int i = ii; i < i_end; i++) {
                float* W_row = &W[i * in_size + jj];
                float sum = 0.0f;
                
                // Unroll inner loop by 4 for better performance
                int j = jj;
                for (; j + 3 < j_end; j += 4) {
                    sum += W_row[j - jj] * x[j] + 
                           W_row[j - jj + 1] * x[j + 1] +
                           W_row[j - jj + 2] * x[j + 2] + 
                           W_row[j - jj + 3] * x[j + 3];
                }
                
                // Handle remaining elements
                for (; j < j_end; j++) {
                    sum += W_row[j - jj] * x[j];
                }
                
                out[i] += sum;
            }
        }
    }
}
#endif

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



