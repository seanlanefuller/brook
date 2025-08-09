import random

# More natural sentence patterns with proper preposition usage
animals = ["cat", "dog", "fox", "squirrel"]
adjectives = ["big", "small", "old", "young"]
sitting_actions = ["sat", "slept"]
moving_actions = ["ran", "walked", "jumped"]

# Proper combinations that make semantic sense
sitting_patterns = [
    "The {adj} {animal} {action} quietly on the {surface}.",
    "A {adj} {animal} {action} peacefully on the {surface}.",
    "The {animal} {action} on the {surface}.",
    "A {animal} {action} on the {surface}.",
]

moving_patterns = [
    "The {adj} {animal} {action} quickly through the {area}.",
    "A {adj} {animal} {action} slowly over the {path}.",
    "The {animal} {action} through the {area}.",
    "A {animal} {action} over the {path}.",
]

# Objects that make sense for sitting ON
surfaces = ["log", "branch", "mat", "grass"]
# Areas that make sense for moving THROUGH
areas = ["forest", "garden", "trees"]
# Paths that make sense for moving OVER
paths = ["path", "bridge", "stream"]

# Simple descriptive sentences
simple_sentences = [
    "The sun shone brightly.",
    "Birds sang sweetly.",
    "Water flowed gently.",
    "Wind blew softly.",
    "Trees grew tall.",
    "Flowers bloomed everywhere.",
    "Animals lived peacefully.",
    "Rain fell quietly."
]

with open("data/sentences.txt", "w") as f:
    # Generate sitting sentences (animals ON surfaces)
    for _ in range(80):
        pattern = random.choice(sitting_patterns)
        sentence = pattern.format(
            adj=random.choice(adjectives),
            animal=random.choice(animals),
            action=random.choice(sitting_actions),
            surface=random.choice(surfaces)
        )
        f.write(sentence + "\n")
    
    # Generate moving sentences (animals THROUGH areas or OVER paths)
    for _ in range(80):
        if random.choice([True, False]):
            # Through areas
            pattern = "The {adj} {animal} {action} through the {area}."
            sentence = pattern.format(
                adj=random.choice(adjectives),
                animal=random.choice(animals),
                action=random.choice(moving_actions),
                area=random.choice(areas)
            )
        else:
            # Over paths
            pattern = "A {adj} {animal} {action} over the {path}."
            sentence = pattern.format(
                adj=random.choice(adjectives),
                animal=random.choice(animals),
                action=random.choice(moving_actions),
                path=random.choice(paths)
            )
        f.write(sentence + "\n")
    
    # Add simple descriptive sentences
    for _ in range(96):
        sentence = random.choice(simple_sentences)
        f.write(sentence + "\n")

        
