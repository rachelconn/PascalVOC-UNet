import tensorflow as tf
from tensorflow.keras import layers, activations

def encoder_block(x, filters):
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    return x

def encoder(x):
    block_1_out = encoder_block(x, 64)
    x = layers.MaxPool2D(strides=2, padding='same')(block_1_out)
    block_2_out = encoder_block(x, 128)
    x = layers.MaxPool2D(strides=2, padding='same')(block_2_out)
    block_3_out = encoder_block(x, 256)
    x = layers.MaxPool2D(strides=2, padding='same')(block_3_out)
    block_4_out = encoder_block(x, 512)
    x = layers.MaxPool2D(strides=2, padding='same')(block_4_out)
    x = encoder_block(x, 1024)
    return (block_1_out, block_2_out, block_3_out, block_4_out), x

def decoder_block(x, encoder_output, filters):
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.Conv2DTranspose(filters, 3, strides=2, padding='same')(x)
    x = tf.concat((x, encoder_output), axis=3)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    return x

def decoder(x, block_outputs):
    x = decoder_block(x, block_outputs[3], 512)
    x = decoder_block(x, block_outputs[2], 256)
    x = decoder_block(x, block_outputs[1], 128)
    x = decoder_block(x, block_outputs[0], 64)
    return x

def UNet(num_classes):
    inputs = tf.keras.Input(shape=(None, None, 3))
    block_outputs, encoded = encoder(inputs)
    decoded = decoder(encoded, block_outputs)
    outputs = layers.Conv2D(num_classes, 1)(decoded)
    outputs = layers.Softmax()(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
