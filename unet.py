""" Project Information """
__author__ = "Aaron Flanagan"
__copyright__ = "NUIG"
__credits__ = "Aaron Flanagan"
__version__ = "1.0.0"
__email__ = "A.flanagan18@nuigalway.com"
__status__ = "Development"

from datetime import datetime

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class UNet:

    def __init__(self, img_size, n_class=2, filter_size=64):
        self.img_size = img_size
        self.n_class = n_class
        self.filter_size = filter_size
        self.model = self.build()

    def build(self):
        # Down-sampling phase
        inputs = keras.Input(shape=self.img_size)

        # 4 contract blocks that have skip connections
        x, skip1 = self.contract_block(inputs, self.filter_size)  # 64
        x, skip2 = self.contract_block(x, self.filter_size * 2)  # 128
        x, skip3 = self.contract_block(x, self.filter_size * 4)  # 216
        x, skip4 = self.contract_block(x, self.filter_size * 8)  # 512

        # final contract block, no skips
        x = self.contract_block(x, self.filter_size * 16, include_pool=False)

        # up-sampling phase
        self.filter_size = self.filter_size * 8  # 512
        x = self.expanse_block(x, skip4, self.filter_size)
        x = self.expanse_block(x, skip3, self.filter_size // 2)
        x = self.expanse_block(x, skip2, self.filter_size // 4)
        x = self.expanse_block(x, skip1, self.filter_size // 8)

        outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="U-Net")
        print("Model is built...")

        return model

    def contract_block(self, inputs, f, include_pool=True):
        x = layers.Conv2D(f, kernel_size=3, kernel_initializer="he_normal", padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(f, kernel_size=3, kernel_initializer="he_normal", padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(rate=0.2)(x)

        if include_pool:
            p = layers.MaxPool2D(pool_size=(2, 2))(x)
            return p, x

        return x

    def expanse_block(self, inputs, skip, f):
        x = layers.Conv2DTranspose(f, kernel_size=(2, 2), strides=2, padding='same')(inputs)

        # handle odd dimensions
        if skip.shape[2] % 2 != 0:
            x = layers.ZeroPadding2D(padding=((0, 0), (0, 1)))(x)  # pad right
        elif skip.shape[1] % 2 != 0:
            x = layers.ZeroPadding2D(padding=((0, 1), (0, 0)))(x)  # pad bottom

        # skip connection
        x = layers.Concatenate()([x, skip])  # default axis -1 (depth)
        x = self.contract_block(x, f, include_pool=False)

        return x

    def compile(self, optimizer="sgd", loss='binary_crossentropy'):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy', 'AUC'])

    def fit(self, X, y, validation_split=0.2, epochs=1, batch_size=None):
        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        early_stopping_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

        return self.model.fit(x=X, y=y, epochs=epochs,
                              validation_split=validation_split,
                              callbacks=[tensorboard_callback, early_stopping_callback],
                              batch_size=batch_size,
                              shuffle=True,
                              validation_batch_size=batch_size)

    def summary(self):
        return self.model.summary()

    def predict(self, X, batch_size=None):
        return self.model.predict(X, batch_size=batch_size)

    def plot(self):
        return keras.utils.plot_model(self.model,
                                      "trained_model_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".png",
                                      show_shapes=True)

    def save(self, model_history, folder="models/"):
        filename = datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"
        self.model.save(folder + filename)
        self.model.save_weights(folder + "weights/model_weights_" + filename)
        np.save(folder + "history/" + filename + '_history.npy', model_history.history)

        return filename

    def evaluate(self, X, y, batch_size=None):
        return self.model.evaluate(X, y, batch_size=batch_size)

    def load(filename):
        return keras.models.load_model("models/" + filename)

    def load_weights(self, model, filename):
        return model.load_weights("models/weights/model_weights_" + filename)
