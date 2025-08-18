// BROOK - A Neural Language Model in C
#include "brook.h"

int main(int argc, char* argv[]) {
    srand(time(NULL));
    init_vocab();
    initialize_weights();
    if (load_model())
	{
		printf("Loaded weights.bin\n");
		set_loaded_weights();
	}
    if (!load_training_data("data/story.txt")) {
        cleanup();
        return 1;
    }
	print_model_info();
    interactive_mode();
    cleanup();
    return 0;
}
