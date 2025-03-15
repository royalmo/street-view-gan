from sys import argv, exit

import os
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

from make_models import NOISE_DIM

this_path = Path(__file__).resolve()
SAVE_DIR = this_path.parent.parent / "images"
os.makedirs(SAVE_DIR, exist_ok=True)

def print_usage_and_exit(exit_code):
    print("Usage:")
    print("  python3 test_models.py world_generator_epoch_50")
    print()
    exit(exit_code)

def generate_and_save_images(model, model_name, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        img = (predictions[i, :, :] + 1) / 2
        plt.subplot(4, 4, i+1)
        plt.imshow(img)  # Rescale back to [0, 1] for display
        plt.axis('off')

        # Saving single image too
        plt.imsave(os.path.join(SAVE_DIR, f"{model_name}_{i+1:02d}.png"), img)

    plt.savefig(os.path.join(SAVE_DIR, f"{model_name}.png"))
    plt.show()

if __name__=="__main__":
    if len(argv) < 2:
        print("Error: no model specified.")
        print_usage_and_exit(1)

    model_path = os.path.join(SAVE_DIR, f"{argv[1]}.h5")
    print(f"Loading model: {model_path}.")
    generator = load_model(model_path)
    num_examples_to_generate = 4*4

    # So we see some nice progress
    tf.random.set_seed(42)

    generate_and_save_images(
        generator,
        argv[1].replace('_generator', ''),
        tf.random.normal([num_examples_to_generate, NOISE_DIM]))
    print(f"Done! Check {SAVE_DIR}")
