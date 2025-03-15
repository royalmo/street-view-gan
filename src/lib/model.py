import tensorflow as tf
from tensorflow.keras import layers

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def build_generator(latent_dim, input_size):
    model = tf.keras.Sequential()

    # Start with a small 10*10 feature map
    model.add(layers.Dense(int(input_size[0]/64 * input_size[1]/64 * 512), use_bias=False, input_shape=(latent_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((int(input_size[0]/64), int(input_size[1]/64), 512)))  # Reshape to 10x10x512

    # Upsample: 10->20
    model.add(layers.Conv2DTranspose(256, kernel_size=5, strides=2, padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # 20->40
    model.add(layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # 40->80
    model.add(layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # 80->160
    model.add(layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # 160->320
    model.add(layers.Conv2DTranspose(32, kernel_size=5, strides=2, padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # 320->640
    model.add(layers.Conv2DTranspose(16, kernel_size=5, strides=2, padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Final resize to RGB
    model.add(layers.Conv2DTranspose(3, kernel_size=5, strides=1, padding="same", activation="tanh"))

    return model

def build_discriminator(image_size):
    model = tf.keras.Sequential()

    # 640x640 → 320x320
    model.add(layers.Conv2D(64, kernel_size=5, strides=2, padding="same", input_shape=(image_size[0], image_size[1], 3)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    # 320x320 → 160x160
    model.add(layers.Conv2D(128, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    # 160x160 → 80x80
    model.add(layers.Conv2D(256, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    # 80x80 → 40x40
    model.add(layers.Conv2D(512, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    # 40x40 → 20x20
    model.add(layers.Conv2D(512, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    # 20x20 → 10x10
    model.add(layers.Conv2D(1024, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    # Flatten and Output
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))  # Probability [0,1]

    return model

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
