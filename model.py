import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from network import UNet

def pad_image(image):
    image_h, image_w = image.shape[0:2]
    y_padding, x_padding = (-(image_h % -16), -(image_w % -16))
    if image_h < 256:
        y_padding = 256 - image_h
    if image_w < 256:
        x_padding = 256 - image_w
    paddings = [[0, y_padding], [0, x_padding], [0, 0]]
    return tf.pad(image, tf.constant(paddings))

def augment_image(image, seed, is_image):
    seed = (seed, seed)
    image = tf.image.stateless_random_crop(image, size=(256, 256, image.shape[2]), seed=seed)
    image = tf.image.stateless_random_flip_left_right(image, seed=seed)
    if is_image:
        image = tf.image.stateless_random_brightness(image, max_delta=0.2, seed=seed)
    return image

class Model():
    def __init__(self, model_params, training_params):
        self.model_params = model_params
        self.training_params = training_params

        self.network = UNet(self.model_params.num_classes)
        self.optimizer = tfa.optimizers.AdamW(
            learning_rate=self.training_params.lr,
            weight_decay=self.training_params.weight_decay,
        )
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.network.compile(
            optimizer=self.optimizer,
            loss=self.loss,
        )
        self.load()

    def process_batch(self, batch, augment, seed=None):
        # Open images
        x_files, y_files = batch
        images, labels, label_masks = [], [], []
        for x_file, y_file in zip(x_files, y_files):
            # Load images
            image = pad_image(np.array(Image.open(x_file.numpy()))) / 255
            label = pad_image(np.array(Image.open(y_file.numpy()))[..., np.newaxis])

            # Augmentation
            if augment:
                image = augment_image(image, seed, True)
                label = augment_image(label, seed, False)

            # Mask ambiguous areas and remove them
            label_mask = tf.where(label == 255, 0., 1.)
            label = tf.where(label != 255, label, 0)

            images.append(image)
            labels.append(label)
            label_masks.append(label_mask)

        images = tf.stack(images)
        labels = tf.stack(labels)
        label_masks = tf.stack(label_masks)

        return images, labels, label_masks

    def train(self, dataset):
        # dataset = dataset.shuffle(100).batch(self.training_params.batch_size).repeat()
        dataset = dataset.batch(self.training_params.batch_size).repeat()
        show_loss_every = 50
        show_prediction_every = float('inf')
        save_every = 5000
        total_loss = 0
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        for i, batch in enumerate(dataset, 1):
            x, y, y_mask = self.process_batch(batch, True, i)
            hist = self.network.fit(
                x,
                y,
                sample_weight=y_mask,
                batch_size=self.training_params.batch_size,
                epochs=1,
                verbose=False,
            )
            total_loss += hist.history['loss'][-1]

            if i % show_loss_every == 0:
                print(f'Finished batch {i}. Average loss: {total_loss / show_loss_every}')
                total_loss = 0

            if i % show_prediction_every == 0:
                ax1.set_title('Image')
                ax1.imshow(x[0])
                ax2.set_title('Ground Truth')
                ax2.imshow(y[0])
                ax3.set_title('Prediction')
                prediction = np.squeeze(np.argmax(self.network(tf.expand_dims(x[0], 0)), axis=-1))
                ax3.imshow(prediction)
                plt.ion()
                plt.draw()
                plt.pause(0.001)

            if i % save_every == 0:
                self.save(i)

            if i == self.training_params.num_batches:
                return

    def save(self, i):
        path = os.path.join('models', f'{self.model_params.name}_{i}', 'model')
        self.network.save_weights(path)
        print(f'Saved model to {path}.')

    def load(self):
        path = os.path.join('models', self.model_params.name, 'model')
        try:
            self.network.load_weights(path)
            print(f'Loaded existing model from {path}')
        except:
            print('Created new model')
            return
