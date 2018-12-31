from keras.layers import Conv2D, Dropout, MaxPool2D, Dense, Flatten, Input, Concatenate, BatchNormalization, Softmax, Reshape, Lambda
from keras.layers.merge import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
import keras.backend as K

def conv_layer(x, num_filters: int, kernel_size: int, dropout: float, max_pool: bool) -> MaxPool2D:
    x = Conv2D(filters=num_filters, kernel_size=kernel_size, padding="same")(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=dropout)(x)
    if max_pool:
        x = MaxPool2D()(x)
    return x

def get_model(map_shape: tuple) -> Model:
    num_classes = 5
    
    maps = Input(shape=map_shape, name="maps")

    x = maps

    x = conv_layer(x=x, num_filters=128, kernel_size=(5,5), dropout=0.25, max_pool=False)
    x = conv_layer(x=x, num_filters=128, kernel_size=(5,5), dropout=0.25, max_pool=False)
    x = conv_layer(x=x, num_filters=128, kernel_size=(3,3), dropout=0.25, max_pool=False)
    x = conv_layer(x=x, num_filters=128, kernel_size=(3,3), dropout=0.25, max_pool=False)
    x = Flatten()(x)
    
    x = Dense(512)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = Dense(num_classes)(x)
    out = Softmax()(x)

    model = Model(maps, out)
    model.compile(optimizer="adam",loss="categorical_crossentropy", metrics=["acc"])

    return model
