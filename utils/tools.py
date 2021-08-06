from tensorflow.keras.preprocessing.image import ImageDataGenerator


def ImageGenerator(
    rescale=1./255,
    zoom_range=0.2,
    validation_split=0.0
):
    # Returns the ImageDataGenerator with some properties
    # See https://keras.io/api/preprocessing/image/
    # Parameters:
    #    rescale (float): Used to rescale input images.
    #    zoom_range (float): Range for random zoom.
    #    validation_split (float): Fraction of images reserved for validation
    #                              (strictly between 0 and 1).
    # Returns:
    #    tf.keras.preprocessing.image.ImageDataGenerator:
    # The string which gets reversed
    return ImageDataGenerator(
        rescale=rescale,
        zoom_range=zoom_range,
        validation_split=validation_split)


def FlowFromDir(
    img_generator,
    path,
    color_mode='rgb',
    batch_size=16,
    target_size=(160, 160),
    class_mode='categorical',
    seed=0,
    shuffle=False,
    subset='training'
):
    # Generate batches of tensor image data with real-time data augmentation.
    # See https://keras.io/api/preprocessing/image/
    # Parameters:
    #    img_generator (tf.keras.preprocessing.image.ImageDataGenerator):
    #                Instance of ImageDataGenerator
    #    path (str): Path to the folder with data
    #    color_mode (str): Color mode of your images, e.g., 'rgb'
    #    batch_size (int): The size of the batch you want your images to be in
    #    target_size (tupe): A tuple (width, height) of integers representing
    #                        the new size of the image
    #    seed (int): Fixing the random seed.
    #    shuffle (bool): If true then the data is randomly shuffled.
    #    subset (str): Label if the data set is for training or validation.
    # Returns:
    #    tensorflow.python.keras.preprocessing.image.DirectoryIterator:
    # The iterable type that allows you to access
    # batches of tensors and their category
    return img_generator.flow_from_directory(
        path,
        color_mode=color_mode,
        batch_size=batch_size,
        target_size=target_size,
        class_mode=class_mode,
        seed=seed,
        shuffle=shuffle,
        subset=subset)
