import os
import time
from pathlib import Path
import tensorflow as tf
from . import model

this_path = Path(__file__).resolve()
SAVE_DIR = this_path.parent.parent.parent / "models"
os.makedirs(SAVE_DIR, exist_ok=True)

# Save function
def save_models(prefix, epoch, generator, discriminator):
    generator.save(os.path.join(SAVE_DIR, f"{prefix}_generator_epoch_{epoch}.h5"))
    discriminator.save(os.path.join(SAVE_DIR, f"{prefix}_discriminator_epoch_{epoch}.h5"))
    print(f"Models saved at epoch {epoch}")

def train(dataset, epochs, save_prefix, generator, discriminator, batch_size, noise_dim, start_at=0):

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(images):
        noise = tf.random.normal([batch_size, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = model.generator_loss(fake_output)
            disc_loss = model.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        if (epoch+1)%10==0 or (epoch+1)%25==0:
            save_models(save_prefix, epoch+1+start_at, generator, discriminator)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

