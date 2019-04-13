import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint
import pickle

# Load the pickle files we created in our load_images_notebook file
X_f = pickle.load(open("X_features.pickle", "rb"))
y_l = pickle.load(open("y_labels.pickle", "rb"))

# Normalize the data
X_f = tf.keras.utils.normalize(X_f, axis=1)

# Construct the model
model = Sequential()

#Conv2D: Start with a convolutional layer
#3x3 window
model.add(Conv2D(64, (3,3), input_shape = X_f.shape[1:])) 
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
# Now we do it all again!
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# For good measure, put the data through a final dense layer at the end
# Need to flatten the data from its 2D form for convolutional to 1D form
# for the dense layer
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(.5))
# Final output layer
model.add(Dense(1))
model.add(Activation('sigmoid'))
# In this instance we will use binary crossentropy since it is Clean vs. Contaminated
model.compile(loss="binary_crossentropy",
             optimizer="adam",
             metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=5, min_lr=0.001)
model.fit(X_f, y_l, batch_size=32, validation_split=0.25, epochs=50, callbacks=[reduce_lr])
