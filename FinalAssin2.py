#!/usr/bin/env python
# coding: utf-8

# In[1]:


''' 
Implementing Feed-forward neural networks with Keras and TensorFlow
a. Import the necessary packages
b. Load the training and testing data (MNIST/CIFAR10)
c. Define the network architecture using Keras
d. Train the model using SGD
e. Evaluate the network
f. Plot the training loss and accuracy
'''


# In[2]:


# a. Import the necessary packages -->
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import random


# In[3]:


# b. Load the training and testing data (MNIST/CIFAR10) -->
mnist = tf.keras.datasets.mnist                          #Importing MNIST dataset
# Splitting it into training and testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255


# In[4]:


# c. Define the network architecture using Keras   -->
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10,activation="softmax")
])

model.summary()


# In[7]:


# d. Train the model using SGD  -->
model.compile(optimizer="sgd",
loss="sparse_categorical_crossentropy",
metrics=['accuracy'])

history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=5)


# In[8]:


# e. Evaluate the network   -->
test_loss,test_acc=model.evaluate(x_test,y_test)
print("Loss=%.3f" %test_loss)
print("Accuracy=%.3f" %test_acc)

n=random.randint(0,9999)
plt.imshow(x_test[n])
plt.show()
predicted_value=model.predict(x_test)
plt.imshow(x_test[n])
plt.show()

print('Predicted value: ', predicted_value[n])


# In[9]:


# f. Plot the training loss and accuracy  -->

#Plotting the training accuracy

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


# In[10]:


#Plotting the training loss  

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

