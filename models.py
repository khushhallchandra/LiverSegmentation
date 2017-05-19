"""models implemented for usage in this project"""
from keras.models import Model
from keras.layers import Input, Convolution2D, BatchNormalization, UpSampling2D
from keras.layers.merge import concatenate


def unet(im_size=(128, 128, 3), output_size=1,
         n_blocks=3, n_filters=8, n_convs=1, filter_size=(3, 3),
         bnorm_decoder=False, bnorm_encoder=False):
    """initializes unet model according to 
    https://pdfs.semanticscholar.org/0704/5f87709d0b7b998794e9fa912c0aba912281.pdf
    """
    input_shape = [im_size[2], im_size[0], im_size[1]]

    input = Input(shape=input_shape)

    x = Convolution2D(
        input_shape=input_shape, filters=n_filters,
        kernel_size=filter_size,data_format='channels_first',
        activation='relu',
        padding='same', name='input_conv')(input)

    encoder_layers = []
    for block in range(n_blocks):
        for conv in range(n_convs):
            if conv == n_convs - 1:
                stride = (2, 2)
            else:
                stride = (1, 1)
            x = Convolution2D(
                filters=n_filters * 2, strides=stride,
                kernel_size=filter_size,
                activation='relu',
                padding='same', data_format='channels_first',
                name='conv' + str(block) + '_' + str(conv) + '_' + str(conv))(x)

        if bnorm_encoder:
            x = BatchNormalization()(x)
        encoder_layers.append(x)
        n_filters *= 2

    encoder_layers = encoder_layers[::-1]

    for block in range(n_blocks):
        for conv in range(n_convs):
            if conv == 0:
                x = concatenate([x, encoder_layers[block]], axis=1)

            x = Convolution2D(
                kernel_size=filter_size,
                filters=n_filters, name='deconv' + str(block) + '_' + str(conv),
                activation='relu',
                padding='same', data_format='channels_first')(x)

        x = UpSampling2D(size=(2, 2), data_format='channels_first')(x)
        if bnorm_decoder:
            x = BatchNormalization()(x)
        n_filters /= 2

    x = Convolution2D(filters=output_size, name='output',
                      kernel_size=filter_size,
                      activation='sigmoid',
                      padding='same', data_format='channels_first')(x)

    unet = Model(inputs=[input], outputs=[x])
    unet.compile(optimizer='adamax',
                 loss='binary_crossentropy', metrics=['accuracy'])

    return unet
