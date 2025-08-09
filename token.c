#include "brook.h"

int get_token_id(const char* word) {
    for (int i = 0; i < vocab_size; i++)
        if (strcmp(vocab[i], word) == 0) return i;
    if (vocab_size < MAX_VOCAB) {
        strncpy(vocab[vocab_size], word, MAX_VOCAB_WORD_LEN - 1);
        vocab[vocab_size][MAX_VOCAB_WORD_LEN - 1] = '\0';
        return vocab_size++;
    }
    unsigned int hash = 0;
    for (int i = 0; word[i]; i++)
        hash = hash * 31 + (unsigned char)word[i];
    return hash % vocab_size;
}

// Helper function to normalize a character
static char normalize_char(char c) {
    if (c >= 'A' && c <= 'Z') return c + ('a' - 'A');
    if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) return c;
    if (c == '.') return '.'; // treat as token
    if (c == '\r' || c == '\n') return '|'; // treat as token
    return ' '; // treat all other as space
}

void tokenize(const char* text) {
    size_t text_len = strlen(text);
    int buf_pos = 0;
    char token_buf[MAX_VOCAB_WORD_LEN];
    int token_len = 0;
    size_t i = 0;

    while (i <= text_len && token_count < MAX_TOKENS) {
        char c = (i < text_len) ? text[i] : ' '; // force flush at end
        char norm = normalize_char(c);

        if (norm == ' ' || norm == '.' || norm == '|') {
            if (token_len > 0) {
                token_buf[token_len] = '\0';
                int id = get_token_id(token_buf);
                tokens[token_count++] = id;
                token_len = 0;
            }
            if ((norm == '.' || norm == '|') && token_count < MAX_TOKENS) {
                token_buf[0] = norm;
                token_buf[1] = '\0';
                int id = get_token_id(token_buf);
                tokens[token_count++] = id;
            }
        } else {
            if (token_len < MAX_VOCAB_WORD_LEN - 1)
                token_buf[token_len++] = norm;
        }
        i++;
    }
}

int get_token_id_existing(const char* word) {
    for (int i = 0; i < vocab_size; i++)
        if (strcmp(vocab[i], word) == 0) return i;
    return -1;
}

void tokenize_user_input(const char* text, int* out_tokens, int* out_count, int max_tokens) {
    int count = 0;
    size_t i = 0;
    size_t text_len = strlen(text);
    char token_buf[MAX_VOCAB_WORD_LEN];
    int token_len = 0;

    while (i <= text_len && count < max_tokens) {
        char c = (i < text_len) ? text[i] : ' '; // force flush at end
        char norm = normalize_char(c);

        if (norm == ' ' || norm == '.' || norm == '|') {
            if (token_len > 0) {
                token_buf[token_len] = '\0';
                int id = get_token_id_existing(token_buf);
                if (id != -1) out_tokens[count++] = id;
                token_len = 0;
            }
            if ((norm == '.' || norm == '|') && count < max_tokens) {
                // treat '.' and '|' as separate tokens
                token_buf[0] = norm;
                token_buf[1] = '\0';
                int id = get_token_id_existing(token_buf);
                if (id != -1) out_tokens[count++] = id;
            }
        } else {
            if (token_len < MAX_VOCAB_WORD_LEN - 1)
                token_buf[token_len++] = norm;
        }
        i++;
    }
    *out_count = count;
}




