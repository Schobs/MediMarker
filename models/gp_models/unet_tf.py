from ast import Tuple
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
from gpflow.utilities import to_default_float


def unet_model_tf(output_channels, input_shape, image_shape, out_dim, batch_size):
    base_model = tf.keras.applications.MobileNetV2(input_shape=image_shape, include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 32x32
        'block_3_expand_relu',   # 16x16
        'block_6_expand_relu',   # 8x8
        'block_13_expand_relu',  # 4x4
        'block_16_project',      # 2x2
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 3),  # 2x2 -> 4x4
        pix2pix.upsample(256, 3),  # 4x4 -> 8x8
        pix2pix.upsample(128, 3),  # 8x8 -> 16x16
        pix2pix.upsample(64, 3),   # 16x16 -> 32x32
    ]

    inputs = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(
                input_shape=input_shape, batch_size=batch_size
            ),
            tf.keras.layers.Reshape(image_shape)
        ]
    )

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same')  # 64x64 -> 128x128

    x = last(x)
    x = tf.keras.layers.Flatten(),
    x = tf.keras.layers.Dense(out_dim, activation="relu"),

    return tf.keras.Model(inputs=inputs, outputs=x)


def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
    conv = tf.keras.layers.Conv2D(n_filters,
                                  3,  # filter size
                                  activation='relu',
                                  padding='same',
                                  kernel_initializer='HeNormal')(inputs)
    conv = tf.keras.layers.Conv2D(n_filters,
                                  3,  # filter size
                                  activation='relu',
                                  padding='same',
                                  kernel_initializer='HeNormal')(conv)

    conv = tf.keras.layers.BatchNormalization()(conv, training=False)
    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
    else:
        next_layer = conv
    skip_connection = conv
    return next_layer, skip_connection


def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32):
    up = tf.keras.layers.Conv2DTranspose(
        n_filters,
        (3, 3),
        strides=(2, 2),
        padding='same')(prev_layer_input)
    merge = tf.keras.layers.concatenate([up, skip_layer_input], axis=3)
    conv = tf.keras.layers.Conv2D(n_filters,
                                  3,
                                  activation='relu',
                                  padding='same',
                                  kernel_initializer='HeNormal')(merge)

    conv = tf.keras.layers.Conv2D(n_filters,
                                  3,
                                  activation='relu',
                                  padding='same',
                                  kernel_initializer='HeNormal')(conv)
    return conv


# def unet(output_channels, input_shape, image_shape, out_dim, batch_size):
#     inputs = tf.keras.Sequential(
#         [
#             tf.keras.layers.InputLayer(
#                 input_shape=input_shape, batch_size=batch_size
#             ),
#             tf.keras.layers.Reshape(image_shape)
#         ]
#     )
#     skips = []
#     x = inputs
#     n_filters = [32, 64, 128, 256]
#     for i in range(4):
#         x, skip = EncoderMiniBlock(x, n_filters=n_filters[i], dropout_prob=0.3, max_pooling=True)
#         skips.append(skip)

#     for i in range(4):
#         x = DecoderMiniBlock(x, skips[-i - 1], n_filters=n_filters[-i - 1])

#     x = tf.keras.layers.Flatten(),
#     x = tf.keras.layers.Dense(out_dim, activation="relu"),


def UNetCompiled(output_channels, input_shape, image_shape, out_dim, batch_size, n_filters=32, n_classes=1):
    """
    Combine both encoder and decoder blocks according to the U-Net research paper
    Return the model as output
    """
    # Input size represent the size of 1 image (the size used for pre-processing)
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.reshape(inputs, [-1] + image_shape)

    # Encoder includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
    # Observe that the filters are increasing as we go deeper into the network which will increasse the # channels of the image
    cblock1 = EncoderMiniBlock(x, n_filters, dropout_prob=0, max_pooling=True)
    cblock2 = EncoderMiniBlock(cblock1[0], n_filters*2, dropout_prob=0, max_pooling=True)
    cblock3 = EncoderMiniBlock(cblock2[0], n_filters*4, dropout_prob=0, max_pooling=True)
    cblock4 = EncoderMiniBlock(cblock3[0], n_filters*8, dropout_prob=0.3, max_pooling=True)
    cblock5 = EncoderMiniBlock(cblock4[0], n_filters*16, dropout_prob=0.3, max_pooling=False)

    # Decoder includes multiple mini blocks with decreasing number of filters
    # Observe the skip connections from the encoder are given as input to the decoder
    # Recall the 2nd output of encoder block was skip connection, hence cblockn[1] is used
    ublock6 = DecoderMiniBlock(cblock5[0], cblock4[1],  n_filters * 8)
    ublock7 = DecoderMiniBlock(ublock6, cblock3[1],  n_filters * 4)
    ublock8 = DecoderMiniBlock(ublock7, cblock2[1],  n_filters * 2)
    ublock9 = DecoderMiniBlock(ublock8, cblock1[1],  n_filters)

    # Complete the model with 1 3x3 convolution layer (Same as the prev Conv Layers)
    # Followed by a 1x1 Conv layer to get the image to the desired size.
    # Observe the number of channels will be equal to number of output classes
    conv9 = tf.keras.layers.Conv2D(n_filters,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer='he_normal')(ublock9)

    conv10 = tf.keras.layers.Conv2D(n_classes, 1, padding='same')(conv9)

    x = tf.keras.layers.Flatten()(conv10)
    x = tf.keras.layers.Dense(out_dim, activation="relu")(x)
    x = tf.keras.layers.Lambda(to_default_float)(x)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model
