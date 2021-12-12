import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from network import UNet

class CategoricalMeanIoU(tf.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, tf.argmax(y_pred, axis=-1), sample_weight)

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
        self.iou_metric = CategoricalMeanIoU(self.model_params.num_classes, name='iou')
        self.network.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=[self.iou_metric],
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

    def train(self, train_ds, validation_ds):
        train_ds = train_ds.batch(self.training_params.batch_size)
        train_ds_size = tf.data.experimental.cardinality(train_ds)
        validation_ds = validation_ds.batch(1)
        save_every = 2
        show_prediction_every = float('inf')
        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        loss_history = []
        val_loss_history = []
        iou_history = []
        val_iou_history = []

        for epoch in range(1, self.training_params.num_epochs + 1):
            total_loss = 0
            total_iou = 0
            for i, batch in enumerate(train_ds, 1):
                x, y, y_mask = self.process_batch(batch, True, i + (epoch * train_ds_size))
                hist = self.network.fit(
                    x,
                    y,
                    sample_weight=y_mask,
                    batch_size=self.training_params.batch_size,
                    epochs=1,
                    verbose=False,
                )
                total_loss += hist.history['loss'][-1]
                total_iou += hist.history['iou'][-1]

                # if i % show_prediction_every == 0:
                #     ax1.set_title('Image')
                #     ax1.imshow(x[0])
                #     ax2.set_title('Ground Truth')
                #     ax2.imshow(y[0])
                #     ax3.set_title('Prediction')
                #     prediction = np.squeeze(np.argmax(self.network(tf.expand_dims(x[0], 0)), axis=-1))
                #     ax3.imshow(prediction)
                #     plt.ion()
                #     plt.draw()
                #     plt.pause(0.001)

            # Epoch done, print out stats
            mean_loss = total_loss / i
            mean_iou = total_iou / i
            print(f'Finished epoch {epoch}.')
            print(f'  Training loss: {mean_loss}')
            print(f'  Training IoU: {mean_iou}')
            loss_history.append(mean_loss)
            iou_history.append(mean_iou)
            total_loss = 0
            total_iou = 0
            # Evaluate on validation set
            val_loss = 0
            val_iou = 0
            for i, batch in enumerate(validation_ds, 1):
                x, y, y_mask = self.process_batch(batch, False, i)
                pred = self.network(x)
                val_loss += self.network.loss(y, pred, y_mask).numpy()
                val_iou += self.iou_metric(y, pred, y_mask).numpy()
            mean_val_loss = val_loss / i
            mean_val_iou = val_iou / i
            val_loss_history.append(mean_val_loss)
            val_iou_history.append(mean_val_iou)
            print(f'  Validation loss: {mean_val_loss}')
            print(f'  Validation IoU: {mean_val_iou}')

            # Save as needed
            if epoch % save_every == 0:
                self.save(epoch)

        # Done training, plot loss/IoU over time
        t = list(str(i) for i in range(1, self.training_params.num_epochs + 1))
        plt.title('Cross-Entropy Loss over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Cross-Entropy Loss')
        plt.plot(t, loss_history, label='Training')
        plt.plot(t, val_loss_history, label='Validation')
        plt.legend()
        plt.show()

        plt.title('Mean IoU over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Mean IoU')
        plt.plot(t, iou_history, label='Training')
        plt.plot(t, val_iou_history, label='Validation')
        plt.legend()
        plt.show()

    def test(self, test_ds):
        test_ds = test_ds.batch(1)
        loss = 0
        iou = 0
        for i, batch in enumerate(test_ds, 1):
            x, y, y_mask = self.process_batch(batch, False, 0)
            pred = self.network(x)
            loss += self.network.loss(y, pred, y_mask).numpy()
            iou += self.iou_metric(y, pred, y_mask).numpy()
        mean_loss = loss / i
        mean_iou = iou / i
        print(f'Testing loss: {mean_loss}')
        print(f'Testing IoU: {mean_iou}')

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
