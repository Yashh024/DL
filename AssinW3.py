#!/usr/bin/env python
# coding: utf-8

# In[12]:


"""
Build the Image classification model by dividing the model into the following fourstages:
a. Loading and preprocessing the image data
b. Defining the model’s architecture
c. Training the model
d. Estimating the model’s performance
"""


# In[13]:


# Importing required packages
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Conv2D, Dropout, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np


# In[14]:


# a. Loading and preprocessing the image data

mnist=tf.keras.datasets.mnist
# Splitting into training and testing data
(x_train,y_train),(x_test,y_test) = mnist.load_data()
input_shape = (28,28,1)


# In[15]:


# Making sure that the values are float so that we can getdecimal points after division
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


# In[16]:


# print("Data type of x_train:", x_train.dtype)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# print("Data type after converting to float:", x_train.dtype)


# In[17]:


# Normalizing the RGB codes by divinding it into the max RGB value.
x_train = x_train / 255
x_test = x_test / 255
print("Shape of training : ", x_train.shape)
print("Shape of testing : ", x_test.shape)


# In[18]:


# b. Defining the model’s architecture
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(200,activation = "relu"))
model.add(Dropout(0.3))

model.add(Dense(10,activation = "softmax"))
model.summary()


# In[19]:


# c. Training the model
model.compile(optimizer="adam",
loss="sparse_categorical_crossentropy",metrics=['accuracy'])
model.fit(x_train,y_train,epochs=2)


# In[20]:


# d. Estimating the model’s performance
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Loss=%.3f" %test_loss)
print("Accuracy=%.3f" %test_acc)


# In[21]:


# Showing image at positions[] from dataset: 
image = x_train[6]
plt.imshow(np.squeeze(image), cmap='gray')
plt.show()


# In[22]:


# Predicting the class of image :
image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
predict_model = model.predict([image])
print("Predicted class {}:" .format(np.argmax(predict_model)))


# In[ ]:




