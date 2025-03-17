# Street View GAN

*A Google Street View Generative Adversarial Network (GAN).*

![gifmaker_me(1)](https://github.com/user-attachments/assets/2fd3596a-069f-4422-b4b7-2de83257c72f)

---

This repository contains the source code of the final project of the
*Advanced DL Modes* course thaught at EPSEM-UPC in the MSc on
*ML and Cybersecurity for Internet-Connected Systems*
([more info](https://epsem.upc.edu/ca/estudis/masters-universitaris/merit)).

Made by [Eric Roy](https://ericroy.net). Licensed under GPL3.

## Overview

This repository contains all the code needed to train and play with
a GAN that generates Street-View images. It can be trained to generate
generic images or from specific coordinates (e.g. France).

I also include some already trained models to just rush into the test part.
The models are trained using Pau Chambaz's
[Google Street View Dataset](https://www.kaggle.com/datasets/paulchambaz/google-street-view).

You will also find the report (pdf TODO LINK) that I submitted for the
MSc course, where I wrote all the juicy details.

## Install steps

First, get yourself with a copy of the repository. You can download the
latest release (TODO LINK) in order to get the trained models and
builded PDF report.

Once inside, **the first time only**, setup venv and install the required packages.
Make sure you have the GPU drivers for Tensorflow
([install steps](https://www.tensorflow.org/install/pip)).

```sh
python3 -m venv tf
source tf/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Then, everytime, before running the code, activate the virtual environment.

```sh
source tf/bin/activate
```

## Train the models (optional)

If you don't want to use the models I pre-trained for you, perform the following
steps to be able to retreive the data, prepare it, train the GAN, and store the
model.

```sh
python3 src/make_models.py
python3 src/make_models.py --region france --epochs 25
# Assuming models/world_{generator,discriminator}_epoch_5.h5 exists
python3 src/make_models.py --region world --epochs 10 --start-at 5
```

The models are stored in /models (only every 10 and 25).

## Play with a model

Once the model is trained (or downloaded) you can play with it on a
CLI-script, so it's easier to run it on a remote server.

```sh
python3 src/test_models.py world_generator_epoch_50
```

This will generate 16 images with seed 42 and store them in /images.
