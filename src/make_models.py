from sys import argv

import kagglehub
import tensorflow as tf

import lib

REGIONS = {
    'world' : None,
    'france': [(51.1, -5.2), (41.3, 4.6)], # [nw corner, se corner]
    'canada': [(83.1, -139.0), (41.7, -97.2)],
    'portugal': [(42.0, -9.6), (38.6, -6.2)],
    'africa': [(37.5, 34.8), (-17.0, 51.4)],
    'europe': [(71, -10), (36, 25)],
    'north_america': [(72, -130), (30, -88)],
    'south_america': [(12.5, -81), (-34.5, -34)],
}

NOISE_DIM = 100  # Dimension of random noise vector
BATCH_SIZE = 32 # 64 returns OOM on the Merit-GPU-Server :O
BUFFER_SIZE = 1000

def get_argv(param, default):
    if f'--{param}' not in argv: return default

    position = argv.index(f'--{param}')
    if position == len(argv): return default # --param is last word

    return argv[position+1]

def within_region(region, coords):
    if region == 'world': return True
    assert region in REGIONS, f"Unknown region! Possible values are: {', '.join(REGIONS.keys())}"

    north, west = REGIONS[region][0]
    south, east = REGIONS[region][1]
    lat, long = coords
    return  lat <= north and lat >= south and long >= west and long <= east

if __name__=="__main__":
    DATASET_PATH = kagglehub.dataset_download("paulchambaz/google-street-view")
    print(f"Dataset location: {DATASET_PATH}")

    COORDS = lib.dataset.retrieve_coords(f"{DATASET_PATH}/dataset/coords.csv")
    DATASET_LENGTH = len(COORDS)
    print(f"Dataset length: {DATASET_LENGTH}")

    # Ensuring same coords
    image_size = lib.dataset.get_ensure_image_size(f"{DATASET_PATH}/dataset", DATASET_LENGTH)
    print(f"Image sizes: {image_size}")

    selected_region = get_argv('region', 'world')
    print(f"Using region region: {selected_region}")
    image_paths = [f"{DATASET_PATH}/dataset/{i}.png" for i, coords in enumerate(COORDS) if within_region(selected_region, coords)]
    print(f"Region dataset length: {len(image_paths)}")

    assert len(image_paths) > 10, "These aren't enough to train the GAN!"
    # Possible improvement be able to use pre-trained model with little images
    assert len(image_paths) > 1000, "These aren't enough to train the GAN!"

    print("Pre-processing data...")
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(lib.dataset.load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    print("Pre-processing done!")

    generator = lib.model.build_generator(NOISE_DIM, lib.dataset.IMAGE_SIZE)
    generator.summary()

    discriminator = lib.model.build_discriminator(lib.dataset.IMAGE_SIZE)
    discriminator.summary()

    print("Starting training")
    lib.train.train(dataset, epochs=50, save_prefix="test_one",
                    generator=generator, discriminator=discriminator,
                    batch_size=BATCH_SIZE, noise_dim=latent_dim)
    print("Training done!")
