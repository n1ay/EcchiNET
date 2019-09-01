from keras.layers import LSTM, Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, TimeDistributed, Flatten, Dropout
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator

IMG_SHAPE = [144, 256, 1]
AUDIO_SHAPE = [2, 1024]

def build_model_video():
    model_video = Sequential([
        Conv2D(input_shape=IMG_SHAPE, filters=16, activation='relu', kernel_size=7,
               padding='valid', strides=(3, 3), data_format='channels_last'),
        MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid', data_format='channels_last'),
        Dropout(0.2),
        Flatten(),
        Dense(units=128, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])
    compile_model(model_video)
    return model_video


def build_model_audio():
    model_audio = Sequential([
        Conv1D(input_shape=AUDIO_SHAPE, filters=16,
               activation='relu', kernel_size=9,
               padding='valid', strides=4, data_format='channels_first'),
        MaxPooling1D(pool_size=2, strides=1, padding='valid', data_format='channels_first'),
        Dropout(0.2),
        Flatten(),
        Dense(units=128, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])
    compile_model(model_audio)
    return model_audio


def build_model_merging():
    model_merging = Sequential([
        Dense(input_shape=[2], units=32, activation='relu'),
        Dense(units=128, activation='relu'),
        Dropout(0.2),
        Dense(units=1, activation='sigmoid')
    ])
    compile_model(model_merging)
    return model_merging


def compile_model(model: Sequential):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


def build_data_generator():
    data_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.4,
        height_shift_range=0.4,
        horizontal_flip=True
    )
    return data_generator
