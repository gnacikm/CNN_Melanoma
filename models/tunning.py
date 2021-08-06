import tensorflow as tf


def build_model_seq(hp, input_shape, num_of_classes):
    # This function builds a sequentional CNN model for the keras tuner.
    # See https://keras.io/keras_tuner/
    # Parameters:
    #    hp: Hyperparameters to be used by keras tuner function
    # Returns:
    #    tf.keras.Model: The CNN with a particular number of layers
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    for i in range(hp.Int('conv_blocks', 1, 4, default=3)):
        filters = hp.Int(f'filters_{i}', 32, 128, step=32)
        model.add(tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            activation='relu'
        ))
        model.add(tf.keras.layers.BatchNormalization())
        if hp.Choice('pooling_' + str(i), ['avg', 'max']) == 'max':
            model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        else:
            model.add(tf.keras.layers.AvgPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(
        hp.Float('dropoutConv', 0., 0.5, step=0.1, default=0.3)))
    model.add(tf.keras.layers.Flatten())
    for j in range(hp.Int('n_layers', 1, 3, default=2)):
        nodes = hp.Int(f'hid_nodes_{j}', 20, 180, step=10, default=60)
        model.add(tf.keras.layers.Dense(nodes, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(
            hp.Float('dropoutDense', 0, 0.5, step=0.1, default=0.3)))
    model.add(tf.keras.layers.Dense(num_of_classes, activation='softmax'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Float(
            'learning_rate',
            min_value=1e-4,
            max_value=1e-2,
            sampling='LOG',
            default=1e-3)),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model
