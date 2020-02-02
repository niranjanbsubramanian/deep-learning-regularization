import numpy as np
import keras.backend as K
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, ReLU
from keras.datasets import fashion_mnist
from keras.optimizers import Adam


#setup parameters 
batch_size = 128
num_classes = 10
epochs = 50

#input dimensions
img_rows, img_cols = 28, 28

#load the data and normalize the inputs 
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
 
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(2,2), padding='same', \
                    input_shape=(X_train.shape[1], X_train.shape[2], 1)))
model.add(ReLU())
model.add(BatchNormalization())
model.add(MaxPooling2D(padding='same'))

model.add(Conv2D(filters=64, kernel_size=(2,2), padding='same'))
model.add(ReLU())
model.add(Dropout(0.5))
model.add(MaxPooling2D(padding='same'))
model.add(Flatten())

model.add(Dense(units=1024))
model.add(ReLU())
model.add(BatchNormalization())

model.add(Dense(units=num_classes))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', \
            optimizer=Adam(lr=0.0001, decay=1e-5), \
            metrics=['accuracy'])

datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, rotation_range=40,zoom_range=0.2, \
                             horizontal_flip=True, fill_mode='nearest')
datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),\
                             validation_data=(X_test, y_test), steps_per_epoch = len(X_train) // batch_size, \
                              epochs=epochs)

score = model.evaluate(X_test, y_test, batch_size=batch_size)
print(score)