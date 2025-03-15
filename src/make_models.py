import os
import kagglehub
import tensorflow as tf

import lib

if __name__=="__main__":
    DATASET_PATH = kagglehub.dataset_download("paulchambaz/google-street-view")
    print(f"Dataset location: {DATASET_PATH}")

    COORDS = lib.dataset.retrieve_coords(f"{DATASET_PATH}/dataset/coords.csv")
    DATASET_LENGTH = len(COORDS)
    print(f"Dataset length: {DATASET_LENGTH}")

    # Ensuring same coords
    print(f"Image sizes: {lib.dataset.get_ensure_image_size(f"{DATASET_PATH}/dataset", DATASET_LENGTH)}")

    BATCH_SIZE = 64
    BUFFER_SIZE = 1000

    print("Pre-processing data...")
    image_paths = [f"{DATASET_PATH}/dataset/{i}.png" for i in range(DATASET_LENGTH)]
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(lib.dataset.load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    print("Pre-processing done!")

    latent_dim = 100  # Dimension of random noise vector
    generator = lib.model.build_generator(latent_dim, lib.dataset.IMAGE_SIZE)
    generator.summary()

    discriminator = lib.model.build_discriminator(lib.dataset.IMAGE_SIZE)
    discriminator.summary()

    print("Starting training")
    lib.train.train(dataset, epochs=50, save_prefix="test_one",
                    generator=generator, discriminator=discriminator,
                    batch_size=BATCH_SIZE, noise_dim=latent_dim)
    print("Training done!")

# TODO be able to use a pre-trained model from the previous time
# TODO care about the coords, for now we don't care
