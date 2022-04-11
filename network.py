import tensorflow as tf
from tensorflow.keras import layers, activations
from tensorflow_examples.models.pix2pix import pix2pix

def encoder_block(x, filters):
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    return x

def encoder(x):
    # TODO: make sure x is in range [0, 255] instead of [0, 1]
    preprocessed = tf.keras.applications.resnet50.preprocess_input(x)
    resnet_encoder = tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        input_tensor=x,
        weights='imagenet',
    )
    resnet_encoder.trainable = False

    encoded = resnet_encoder(preprocessed)
    # Last layer with full resolution
    block_1_out = resnet_encoder.get_layer('input_1').output
    # Last layer with 1/2 resolution
    block_2_out = resnet_encoder.get_layer('conv1_relu').output
    # Last layer with 1/4 resolution
    block_3_out = resnet_encoder.get_layer('conv2_block3_out').output
    # Last layer with 1/8 resolution
    block_4_out = resnet_encoder.get_layer('conv3_block4_out').output
    # Last layer with 1/16 resolution
    block_5_out = resnet_encoder.get_layer('conv4_block6_out').output

    return (block_1_out, block_2_out, block_3_out, block_4_out, block_5_out), encoded

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
    x = decoder_block(x, block_outputs[4], 1024)
    x = decoder_block(x, block_outputs[3], 512)
    x = decoder_block(x, block_outputs[2], 256)
    x = decoder_block(x, block_outputs[1], 64)
    x = decoder_block(x, block_outputs[0], 3)
    return x

def UNet(num_classes):
    # inputs = tf.keras.Input(shape=(None, None, 3))
    # block_outputs, encoded = encoder(inputs)
    # decoded = decoder(encoded, block_outputs)
    # outputs = layers.Conv2D(num_classes, 1)(decoded)
    # outputs = layers.Softmax()(outputs)
    # model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # return model

    """
        Lovingly lifted from:
        https://www.tensorflow.org/tutorials/images/segmentation
    """
    # TODO: need to use 224px square crops for input
    input_shape = (224, 224, 3)
    inputs = tf.keras.Input(shape=input_shape)

    encoder = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
    )
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    encoder_outputs = [encoder.get_layer(name).output for name in layer_names]
    encoder = tf.keras.Model(inputs=encoder.input, outputs=encoder_outputs)
    encoder.trainable = False

    encoded = encoder(inputs)
    x = encoded[-1]
    encoder_layers = reversed(encoded[:-1])
    decoder_layers = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]
    outputs = decoder_layers[0](x)

    for decoder_layer, encoder_layer in zip(decoder_layers, encoder_layers):
        x = decoder_layer(x)
        x = layers.Concatenate()([x, encoder_layer])

    outputs = tf.keras.layers.Conv2DTranspose(
        filters=num_classes,
        kernel_size=3,
        strides=2,
        padding='same'
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)
