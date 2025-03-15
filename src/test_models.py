from sys import argv, exit

import os
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

from make_models import NOISE_DIM
import lib

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

    plt.figure(figsize=(12, 12), dpi=100)
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow((predictions[i, :, :] + 1) / 2) # Rescale back to [0, 1] for display
        plt.axis('off')
    plt.savefig(os.path.join(SAVE_DIR, f"{model_name}.png"),bbox_inches='tight')

    # Save individual images (i don't know why doing it with the full image
    # causes some weird output)
    for i in range(predictions.shape[0]):
        plt.figure(figsize=(9, 9), dpi=100)  # 640 px / 100 dpi = 6.4 inches
        # plt.subplot(1, 1, 1)
        plt.imshow((predictions[i, :, :] + 1) / 2)
        plt.axis('off')
        plt.savefig(os.path.join(SAVE_DIR, f"{model_name}_{i+1:02d}.png"),
                    bbox_inches='tight',  # Remove extra whitespace
                    pad_inches=0) # No padding around the image

if __name__=="__main__":
    if len(argv) < 2:
        print("Error: no model specified.")
        print_usage_and_exit(1)

    model_path = os.path.join(lib.train.SAVE_DIR, f"{argv[1]}.h5")
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
