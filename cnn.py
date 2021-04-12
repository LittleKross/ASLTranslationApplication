import os
import cv2

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

path = './frames'
# iterating over each image
count = 0
for img in os.listdir(path):
    # convert to array
    img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)

    plt.imshow(img_array, cmap='gray')
    plt.show()
    
    count += 1
    if count == 5:
        break

# model = keras.Sequential([
#     keras.layers.Conv2D(64, 3, activation='relu', input_shape=()),
#     keras.layers.Conv2D(32, 3, activation='relu'),
#     keras.layers.Flatten(),
#     keras.layers.Dense(3, activation='softmax')
# ])


# # define a loss function
# model.compile(optimizer='adam',
#               loss=keras.losses.SparseCategoricalCrossentropy(),
#               metrics=['accuracy'])

# model.fit(images, labels, epochs=5, batch_size=32)

# model.evaluate(test_images, test_labels)

# model = keras.Sequential([
#    keras.layers.Conv2D(64, 3, activation='relu', input_shape=(300,300,1)),
#    keras.layers.Conv2D(32, 3, activation='relu'),
#    keras.layers.Flatten(),
#    keras.layers.Dense(3, activation='softmax')
# ])

# model.compile(optimizer='adam',
#               loss=keras.losses.SparseCategoricalCrossentropy(),
#               metrics=['accuracy'])

# model.fit(images, labels, epochs=5, batch_size=32)

# model.evaluate(test_images, test_labels)

# model = keras.Sequential([
#    keras.layers.AveragePooling2D(6,3, input_shape=(300,300,1)),
#    keras.layers.Conv2D(64, 3, activation='relu'),
#    keras.layers.Conv2D(32, 3, activation='relu'),
#    keras.layers.MaxPool2D(2,2),
#    keras.layers.Dropout(0.5),
#    keras.layers.Flatten(),
#    keras.layers.Dense(128, activation='relu'),
#    keras.layers.Dense(3, activation='softmax')
# ])

# model.compile(optimizer='adam',
#               loss=keras.losses.SparseCategoricalCrossentropy(),
#               metrics=['accuracy'])

# model.fit(images, labels, epochs=5, batch_size=32)

# model.evaluate(test_images,test_labels)

# from kerastuner.tuners import RandomSearch

# def build_model(hp):
#   model = keras.Sequential()

#   model.add(keras.layers.AveragePooling2D(6,3,input_shape=(300,300,1)))

#   for i in range(hp.Int("Conv Layers", min_value=0, max_value=3)):
#     model.add(keras.layers.Conv2D(hp.Choice(f"layer_{i}_filters", [16,32,64]), 3, activation='relu'))
  
#   model.add(keras.layers.MaxPool2D(2,2))
#   model.add(keras.layers.Dropout(0.5))
#   model.add(keras.layers.Flatten())

#   model.add(keras.layers.Dense(hp.Choice("Dense layer", [64, 128, 256, 512, 1024]), activation='relu'))

#   model.add(keras.layers.Dense(3, activation='softmax'))

#   model.compile(optimizer='adam',
#               loss=keras.losses.SparseCategoricalCrossentropy(),
#               metrics=['accuracy'])
  
#   return model