 BROOK - A Configurable Neural Language Model in C
 
 Named for its flowing, natural way of connecting words and concepts,
 like a brook that connects different parts of the landscape with
 gentle, continuous flow.
 
 This is a feedforward neural network that learns to predict the next word
 in a sequence. It uses word embeddings, multiple hidden layers with ReLU
 activation, dropout regularization, and various optimization techniques.
 
 Architecture Overview:
 Input → Word Embeddings → Hidden Layer 1 → ... → Hidden Layer N → Output
 
 Key Features:
 - Configurable number of hidden layers
 - Dynamic memory allocation for different architectures
 - Xavier weight initialization for stable training
 - Gradient clipping to prevent exploding gradients
 - Top-k sampling for diverse text generation
 - Interactive training and generation interface

Commands:

  train - train with default num epochs

  train N - train with N epochs

  save - save current model

  vocab - list all vocabulary words

  tokens - list some tokens

  quit - exit

  or type seed words for text generation
