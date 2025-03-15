
import tensorflow as tf
import random
from PIL import Image

IMAGE_SIZE = None

def retrieve_coords(csv):
    with open(csv) as coords_file:
        coords = [[float(x) for x in line.split(',')] for line in coords_file.readlines()]
    return coords

def get_ensure_image_size(basepath, dataset_length):
    global IMAGE_SIZE
    with Image.open(f"{basepath}/0.png") as img:  
        image_size = img.size # (width, height)

    for _ in range(100):
        with Image.open(f"{basepath}/{random.randint(0, dataset_length-1)}.png") as img:
            assert image_size == img.size

    IMAGE_SIZE = image_size
    return image_size

# Function to load and preprocess images
def load_and_preprocess(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)  # Decode as RGB
    img = tf.image.convert_image_dtype(img, tf.float32)  # Normalize [0,1]

    # Apply data augmentation
    img = tf.image.random_brightness(img, max_delta=0.2)  # Adjust brightness
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)  # Adjust contrast

    # Random zoom (crop and resize)
    scale = tf.random.uniform([], 0.9, 1.0)  # Zoom from 90% to 100%
    new_size = tf.cast(scale * IMAGE_SIZE[0], tf.int32)
    img = tf.image.resize_with_crop_or_pad(img, new_size, new_size)  # Crop
    img = tf.image.resize(img, IMAGE_SIZE)  # Resize back

    # Normalize to [-1,1] for DCGAN
    img = (img - 0.5) * 2
    return img
