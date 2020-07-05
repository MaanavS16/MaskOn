import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# augment training data
train_datagen = ImageDataGenerator(rescale=1/255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# don't augment validation data
val_datagen = ImageDataGenerator(rescale=1/255)

# load training data
train_generator = train_datagen.flow_from_directory(
        str(os.getcwd()) + '/maskData/train',
        target_size = (256, 256),
        batch_size = 40,
        class_mode = 'binary')

# load validation data
val_generator = val_datagen.flow_from_directory(
        str(os.getcwd()) + '/maskData/val',
        target_size = (256, 256),
        batch_size = 40,
        class_mode = 'binary')

# define keras model
model = tf.keras.Sequential([tf.keras.layers.Conv2D(16, (3,3), input_shape = (256, 256, 3), activation='relu'),
                             tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                             tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
                             tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                             tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                             tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                             tf.keras.layers.Dropout(0.3),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(256, activation = 'relu'),
                             tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dense(1, activation = 'sigmoid')])

# compile and print model summary
model.compile(loss ="binary_crossentropy", optimizer='adam', metrics=['acc'])
model.summary()

# train and store accuracy and loss over time
history = model.fit(train_generator, epochs=1, validation_data = val_generator)

'''
# load and reshape image for testing
testPath = ''
img = image.load_img(testPath, target_size=(256, 256))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# make and print predictions
images = np.vstack([x])
preds = model.predict(images)
print(preds)
'''
