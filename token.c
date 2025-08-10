#include "brook.h"

/**
 * Looks up a word in the vocab, optionally adding it if not found.
 * Returns token id or -1 if not found (and add_new==0).
 */
int get_token_id_common(const char* word, int add_new) {
    for (size_t i = 0; i < vocab_size; i++)
        if (strcmp(vocab[i], word) == 0) return i;
    if (add_new && vocab_size < MAX_VOCAB) {
        strncpy(vocab[vocab_size], word, MAX_VOCAB_WORD_LEN - 1);
        vocab[vocab_size][MAX_VOCAB_WORD_LEN - 1] = '\0';
        return vocab_size++;
    }
    if (add_new) {
        unsigned int hash = 0;
        for (size_t i = 0; word[i]; i++)
            hash = hash * 31 + (unsigned char)word[i];
        return hash % vocab_size;
    }
    return -1;
}

/**
 * Normalizes a character for tokenization.
 * Converts to lowercase, maps sentence/sequence ends, and spaces.
 */
static char normalize_char(char c) {
    if (c >= 'A' && c <= 'Z') return c + ('a' - 'A');
    if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) return c;
    if (c == '.') return '.'; // end of sentence token
    if (c == '\r' || c == '\n') return '|'; // end of sequence token
    return ' '; // treat all other as space
}

typedef int (*token_lookup_fn)(const char*);

int token_lookup_add(const char* word) {
    return get_token_id_common(word, 1);
}
int token_lookup_existing(const char* word) {
    return get_token_id_common(word, 0);
}

/**
 * Generic tokenization function.
 * Normalizes input, splits into tokens, and looks up token ids.
 * If out_count is not NULL, sets the number of tokens found.
 */
static void tokenize_generic(const char* text, int* out_tokens, int* out_count, int max_tokens, token_lookup_fn lookup) {
    int count = 0;
    size_t i = 0;
    size_t text_len = strlen(text);
    char token_buf[MAX_VOCAB_WORD_LEN];
    int token_len = 0;

    while (i <= text_len && count < max_tokens) {
        char c = (i < text_len) ? text[i] : ' ';
        char norm = normalize_char(c);

        if (norm == ' ' || norm == '.' || norm == '|') {
            if (token_len > 0) {
                token_buf[token_len] = '\0';
                int id = lookup(token_buf);
                if (id != -1) out_tokens[count++] = id;
                token_len = 0;
            }
            if ((norm == '.' || norm == '|') && count < max_tokens) {
                token_buf[0] = norm;
                token_buf[1] = '\0';
                int id = lookup(token_buf);
                if (id != -1) out_tokens[count++] = id;
            }
        } else {
            if (token_len < MAX_VOCAB_WORD_LEN - 1)
                token_buf[token_len++] = norm;
        }
        i++;
    }
    if (out_count) *out_count = count;
}

/**
 * Tokenizes input text, adding new words to vocab.
 * Uses global tokens/token_count.
 */
void tokenize(const char* text) {
    tokenize_generic(text, tokens, &token_count, MAX_TOKENS, token_lookup_add);
}

/**
 * Tokenizes user input, only using existing vocab.
 */
void tokenize_user_input(const char* text, int* out_tokens, int* out_count, int max_tokens) {
    tokenize_generic(text, out_tokens, out_count, max_tokens, token_lookup_existing);
}




