import tensorflow as tf


class LeNet(tf.keras.Model):
    """
    Creates a simple CNN in tensorflow.keras using the Functional API
    See https://keras.io/guides/functional_api/
    See also https://keras.io/api/models/model/

    This CNN has a similar architecture to LeNet-5 introduced by LeCun et al in
    https://doi.org/10.1162%2Fneco.1989.1.4.541

    Extends `tf.keras.Model`.
    """

    def __init__(self, num_of_classes):
        super(LeNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu'
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation='relu',
        )
        self.maxpooling = tf.keras.layers.MaxPooling2D()
        self.avgpooling = tf.keras.layers.AveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=120, activation="relu")
        self.dense2 = tf.keras.layers.Dense(units=84, activation="relu")
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.dense_out = tf.keras.layers.Dense(
            units=num_of_classes,
            activation="softmax")

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.maxpooling(x)
        x = self.conv2(x)
        x = self.avgpooling(x)
        x = self.flatten(x)
        for layer in [self.dense1, self.dense2]:
            x = layer(x)
            if training:
                x = self.dropout(x, training=training)
        x = self.dense_out(x)
        return x
