from sys import argv, exit
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

from make_models import NOISE_DIM

def print_usage_and_exit(exit_code):
    print("Usage:")
    print("  python3 test_models.py path_to_generator_model.h5")
    print()
    exit(exit_code)

def generate_and_save_images(model, fig_name, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    plt.figure(figsize=(8, 8))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow((predictions[i, :, :] + 1) / 2)  # Rescale back to [0, 1] for display
        plt.axis('off')

    plt.savefig(fig_name)
    plt.show()

if __name__=="__main__":
    if len(argv) < 2:
        print("Error: no model specified.")
        print_usage_and_exit(1)

    model_path = argv[1]
    generator = load_model(model_path)
    num_examples_to_generate = 8*8

    generate_and_save_images(
        generator,
        model_path.replace('.h5', '.png'),
        tf.random.normal([num_examples_to_generate, NOISE_DIM]))
    print("Done!")
