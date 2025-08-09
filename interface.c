#include "brook.h"

void generate_text_from_seed(const char* seed_string, int steps) {
    char buffer[1024];
    strncpy(buffer, seed_string, sizeof(buffer));
    buffer[sizeof(buffer) - 1] = '\0';
    to_lowercase(buffer);

    int context[MAX_CONTEXT], context_len = 0;
    tokenize_user_input(buffer, context, &context_len, MAX_CONTEXT);

    int last_token = -1;
    int words_in_sentence = context_len;
    for (int i = 0; i < steps; i++) {
        int next = predict(context, context_len);
        
        // Anti-repetition: skip if same as last token or if we've seen this pattern recently
        if (next == last_token) continue;
        
        // Check for immediate 2-token loops
        if (context_len >= 2 && next == context[context_len-2] && context[context_len-1] == last_token) {
            continue;  // Skip A-B-A-B patterns
        }
        
        if (strcmp(vocab[next], ".") == 0) {
            if (i != 0) printf(". ");
        } else {
            if (last_token != -1) printf(" ");
            printf("%s", vocab[next]);
        }
        words_in_sentence++;
        if (context_len == context_window) {
            for (int j = 0; j < context_len - 1; j++)
                context[j] = context[j + 1];
            context[context_len - 1] = next;
        } else {
            context[context_len++] = next;
        }
        last_token = next;
    }
    if (words_in_sentence > 0 && last_token != get_token_id(".")) printf(".");
}

void interactive_mode() {
    char input[256] = {0};
    while (1) {
        printf("> ");
        fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;
        input[strcspn(input, "\r\n")] = 0;
        if (strcmp(input, "quit") == 0 || strcmp(input, "exit") == 0
			|| strcmp(input, "q") == 0) {
            break;
		} else if (strcmp(input, "tokens") == 0) {
			printf("first 20 tokens: ");
			for (int i = 0; i < token_count && i < 20; i++) {
				printf("%s ", vocab[tokens[i]]);
			}
        } else if (strcmp(input, "train") == 0) {
            train(context_window, EPOCHS);
            continue;
        } else if (strncmp(input, "train ", 6) == 0) {
            int custom_epochs = atoi(input + 6);
            if (custom_epochs > 0 && custom_epochs <= MAX_EPOCHS) {
                printf("Training %d-layer network with %d epochs and context window %d...\n",
                       num_hidden_layers, custom_epochs, context_window);
                train(context_window, custom_epochs);
            } else {
                printf("Invalid epoch count. Use 1-%d epochs\n", MAX_EPOCHS);
            }
            continue;
        } else if (strcmp(input, "save") == 0) {
            save_model();
            continue;
        } else if (strcmp(input, "vocab") == 0) {
            printf("Vocabulary (%d words):\n", vocab_size);
            for (int i = 0; i < vocab_size; i++)
                printf("%s ", vocab[i]);
            printf("\n");
            continue;
        } else if (strlen(input) == 0) continue;
        else generate_text_from_seed(input, 32);
        printf("\n");
    }
}
