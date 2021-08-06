import tensorflow as tf


class DeepNet(tf.keras.Model):
    """
    Creates a deep CNN in tensorflow.keras using the Functional API
    See https://keras.io/guides/functional_api/
    See also https://keras.io/api/models/model/

    This CNN inherits the bottom layers from a basemodel network,
    for example, one may use one of the deep CNNs from
    https://keras.io/api/applications/
    The purpose of this model is to use transfer learning and train
    only the densely-connected layers.

    Extends `tf.keras.Model`.
    """

    def __init__(self, base_model, num_of_classes):
        super(DeepNet, self).__init__()
        self.basemodel = base_model
        self.globavg = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(units=120, activation="relu")
        self.dense2 = tf.keras.layers.Dense(units=84, activation="relu")
        self.dense_out = tf.keras.layers.Dense(
            units=num_of_classes,
            activation="softmax")
        self.dropout = tf.keras.layers.Dropout(0.3)

    def call(self, inputs, training=False):
        x = self.basemodel(inputs, training=False)
        x = self.globavg(x)
        for layer in [self.dense1, self.dense2]:
            x = layer(x)
            if training:
                x = self.dropout(x, training=training)
        x = self.dense_out(x)
        return x
